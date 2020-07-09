"""Demo for a single image inference"""
import argparse
import logging
import cv2
import matplotlib.pyplot as plt
import time
import datetime

import torch
import torch.distributed as dist
import torchvision

import data
import transforms
import encoder
import models
import logs

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    import apex.optimizers as apex_optim
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example.")

LOG = logging.getLogger(__name__)


def demo_cli():
    parser = argparse.ArgumentParser(
        # __doc__: current module's annotation (or module.a_function's annotation)
        description=__doc__,
        # --help text with default values, e.g., gaussian threshold (default: 0.1)
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    logs.cli(parser)
    data.data_cli(parser)
    models.net_cli(parser)
    encoder.encoder_cli(parser)

    parser.add_argument('--resume', '-r', action='store_true', default=False,
                        help='resume from checkpoint')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--checkpoint-path', '-p',
                        default='link2checkpoints_storage',
                        help='folder path checkpoint storage of the whole pose estimation model')

    group = parser.add_argument_group('apex configuration')
    group.add_argument("--local_rank", default=0, type=int)
    group.add_argument('--opt-level', type=str, default='O1')
    group.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    group.add_argument('--loss-scale', type=str, default=None)  # '1.0'
    group.add_argument('--channels-last', type=bool,
                       default=False)  # fixme: channel last may lead to 22% speed up
    group.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                       help='print frequency (default: 10)')

    args = parser.parse_args(
        '--checkpoint-whole link2checkpoints_storage/PoseNet_3_epoch.pth --resume --no-pretrain'.split())

    return args


def main():
    global best_loss, args
    best_loss = float('inf')
    start_epoch = 0
    args = demo_cli()

    print(f"\nopt_level = {args.opt_level}")
    print(f"keep_batchnorm_fp32 = {args.keep_batchnorm_fp32}")
    print(f"CUDNN VERSION: {torch.backends.cudnn.version()}\n")

    torch.backends.cudnn.benchmark = True
    # print(vars(args))
    args.pin_memory = False
    if torch.cuda.is_available():
        args.pin_memory = True

    args.world_size = 1

    preprocess_transformations = [
        transforms.NormalizeAnnotations(),
        transforms.WarpAffineTransforms(args.square_length, aug_params=args,
                                        debug_show=args.debug_affine_show),
        # transforms.RandomApply(transforms.AnnotationJitter(), 0.1),
    ]

    preprocess_transformations += [
        # transforms.RandomApply(transforms.JpegCompression(), 0.1),
        transforms.RandomApply(transforms.ColorTint(), 0.12),
        transforms.ImageTransform(torchvision.transforms.ToTensor()),
        transforms.ImageTransform(
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])),
    ]
    preprocess = transforms.Compose(preprocess_transformations)

    target_transform = encoder.encoder_factory(args, args.strides)

    train_data, val_data = data.dataset_factory(args, preprocess,
                                                target_transform)

    model, lossfuns = models.model_factory(args)

    model.cuda()

    if args.resume:
        model, _, start_epoch, best_loss = models.networks.load_model(
            model, args.checkpoint_whole, optimizer=None, resume_optimizer=False,
            drop_layers=False)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interpretation with argparse.
    model = amp.initialize(model,
                           opt_level=args.opt_level,
                           keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                           loss_scale=args.loss_scale)


    # 创建数据加载器，在训练和验证步骤中喂数据
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.loader_workers,
                                               pin_memory=args.pin_memory,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.loader_workers,
                                             pin_memory=args.pin_memory,
                                             drop_last=True)

    # ############################# Train and Validate #############################
    for epoch in range(start_epoch, start_epoch + args.epochs):
        test(train_loader, model, lossfuns, epoch)


def test(val_loader, model, criterion, epoch, show_image=True):
    print('\n ############################# Test phase, Epoch: {} #############################'.format(epoch))
    model.eval()
    # DistributedSampler 中记录目前的 epoch 数， 因为采样器是根据 epoch 来决定如何打乱分配数据进各个进程
    # if args.distributed:
    #     val_sampler.set_epoch(epoch)  # 验证集太小，不够4个划分
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, (images, annos, metas) in enumerate(val_loader):

        images = images.cuda(non_blocking=True)
        anno_heads = [[x.cuda(non_blocking=True) for x in pack] for pack in
                      annos]

        with torch.no_grad():
            outputs = model(images)  # outputs of multiple headnets
            multi_losses = []
            for out, lossfun, anno in zip(outputs, criterion, anno_heads):
                multi_losses += list(lossfun(out, *anno))
            # weight the multi-task losses
            weighted_losses = [torch.mul(lam, l) for lam, l in
                               zip(args.lambdas, multi_losses)]
            loss = sum(weighted_losses)  # args.lambdas defined in models.factory

        LOG.info({
            'type': f'validate-at-rank{args.local_rank}',
            'epoch': epoch,
            'batch': batch_idx,
            'head_losses': [round(to_python_float(l.detach()) if torch.is_tensor(l) else l, 6)
                            for l in multi_losses],
            'loss': round(to_python_float(loss.detach()), 6),
        })

        if show_image:
            import numpy as np
            hmps = outputs[0][0][1].cpu().numpy()
            offs = outputs[1][0][1].cpu().numpy()
            offs[np.isinf(offs)] = 0
            image = images.cpu().numpy()[0, ...].transpose((1, 2, 0))  # the first image
            show_labels = cv2.resize(hmps[0].transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_CUBIC)
            # offsets = cv2.resize(offsets.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_NEAREST)
            # mask_miss = np.repeat(mask_miss.transpose((1, 2, 0)), 3, axis=2)
            # mask_miss = cv2.resize(mask_miss, image.shape[:2], interpolation=cv2.INTER_NEAREST)
            # image = cv2.resize(image, mask_miss.shape[:2], interpolation=cv2.INTER_NEAREST)
            plt.imshow(image)  # We manually set Opencv earlier: RGB
            # plt.imshow(mask_miss, alpha=0.5)  # mask_all
            plt.imshow(show_labels[:, :, 15], alpha=0.5)  # mask_all
            plt.show()

        if batch_idx % args.print_freq == 0:
            reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), images.size(0))  # update needs average and number
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            print('==================> Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Loss {loss.val:.10f} ({loss.avg:.4f}) <================ \t'.format(
                epoch, batch_idx, len(val_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time,
                loss=losses))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    log_level = logging.INFO  # logging.DEBUG
    # set RootLogger
    logging.basicConfig(level=log_level)

    main()
