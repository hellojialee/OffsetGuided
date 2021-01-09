"""Demo for a single image inference"""
import argparse
import logging
import cv2
import matplotlib
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import multiprocessing
import torch
import torch.distributed as dist
import torchvision

import data
import config
import transforms
import encoder
import models
import logs
from visualization import show
import decoder
from utils.util import AverageMeter

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
    decoder.decoder_cli(parser)

    parser.add_argument('--resume', '-r', action='store_true', default=False,
                        help='resume from checkpoint')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--checkpoint-path', '-p',
                        default='link2checkpoints_storage',
                        help='folder path checkpoint storage of the whole pose estimation model')
    parser.add_argument('--show-limb-idx', default=None, type=int, metavar='N',
                        help='draw the vector of limb connection offset and connected keypoints')
    parser.add_argument('--show-hmp-idx', default=None, type=int, metavar='N',
                        help='show the heatmap and locations of keypoints of current type')
    parser.add_argument('--show-all-limbs', action='store_true', default=False,
                        help='show all candidate limb connections')
    parser.add_argument('--show-detected-poses', action='store_true', default=False,
                        help='show the final results')

    group = parser.add_argument_group('apex configuration')
    group.add_argument("--local_rank", default=0, type=int)
    group.add_argument('--opt-level', type=str, default='O1')
    group.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    group.add_argument('--loss-scale', type=str, default=None)  # '1.0'
    group.add_argument('--channels-last', default=False, action='store_true',
                       help='channel last may lead to 22% speed up')  # fixme: channel last may lead to 22% speed up
    group.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                       help='print frequency (default: 10)')

    # args = parser.parse_args(
    #     '--checkpoint-whole link2checkpoints_storage/PoseNet_18_epoch.pth --resume --no-pretrain'.split())
    args = parser.parse_args()
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
        transforms.WarpAffineTransforms(args.square_length,
                                        aug_params=transforms.FixedAugParams(),
                                        debug_show=args.debug_affine_show),
        # transforms.RandomApply(transforms.AnnotationJitter(), 0.1),
    ]

    preprocess_transformations += [
        # transforms.RandomApply(transforms.JpegCompression(), 0.1),
        transforms.RandomApply(transforms.ColorTint(), 0),
        transforms.ImageTransform(torchvision.transforms.ToTensor()),
        transforms.ImageTransform(
            torchvision.transforms.Normalize(mean=config.data_mean,
                                             std=config.data_std)),
    ]
    preprocess = transforms.Compose(preprocess_transformations)

    target_transform = encoder.encoder_factory(args, args.strides)

    train_loader, val_loader = data.dataloader_factory(args, preprocess,
                                                       target_transform)

    model, lossfuns = models.model_factory(args)

    model.cuda()

    if args.resume:
        model, _, start_epoch, best_loss, _ = models.networks.load_model(
            model, args.checkpoint_whole, optimizer=None, resume_optimizer=False,
            drop_layers=False, load_amp=False)

    processor = decoder.decoder_factory(args)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interpretation with argparse.
    # make processor as Module and wrap it 测试后发现没有加速效果
    [model] = amp.initialize([model],  # , processor
                             opt_level=args.opt_level,
                             keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                             loss_scale=args.loss_scale)

    # ############################# Train and Validate #############################
    for epoch in range(start_epoch, start_epoch + args.epochs):
        test(val_loader, model, lossfuns, epoch, processor)


def test(val_loader, model, criterion, epoch, processor):
    """
    Args:
        val_loader:
        model:
        criterion:
        epoch:
    """
    print('\n ======================= Test phase, Epoch: {} ======================='.format(epoch))
    model.eval()
    # if args.distributed:
    #     val_sampler.set_epoch(epoch)
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

        # post-processing for generating individual poses
        # debug_hmp_output = outputs[0][0][1].cpu().numpy()
        # debug_omp_output = outputs[1][0][1].cpu().numpy()

        batch_poses = processor.generate_poses(outputs)

        if args.show_detected_poses:
            image_poses = batch_poses[0]

            image = images.cpu().numpy()[0, ...].transpose((1, 2, 0))  # the first image
            image = np.clip((image + 2.0) / 4.0, 0.0, 1.0)
            skeleton = encoder.OffsetMaps.skeleton
            keypoint_painter = show.KeypointPainter(
                show_box=False,
                # color_connections=True,
                linewidth=2.5,
                solid_threshold=0.06,  # limbs below this response are drawn with dashed lines
            )
            with show.image_canvas(image,
                                   # output_path + '.keypoints.png',
                                   show=True,
                                   # fig_width=args.figure_width,
                                   # dpi_factor=args.dpi_factor
                                   ) as ax:
                keypoint_painter.keypoints(ax, image_poses[:, :, :3], skeleton=skeleton)

        LOG.info({
            'type': f'validate-at-rank{args.local_rank}',
            'epoch': epoch,
            'batch': batch_idx,
            'head_losses': [round(to_python_float(l.detach()) if torch.is_tensor(l) else l, 6)
                            for l in multi_losses],
            'loss': round(to_python_float(loss.detach()), 6),
        })

        if isinstance(args.show_hmp_idx, int):
            hmps = outputs[0][0][1]
            offs = outputs[1][0][1]

            sizeHW = (args.square_length, args.square_length)
            hmps = torch.nn.functional.interpolate(hmps, size=sizeHW, mode="bicubic")
            offs = torch.nn.functional.interpolate(offs, size=sizeHW, mode="bicubic")
            plt.imshow(hmps.cpu().numpy()[0, 0])
            plt.show()

            filter_map = decoder.hmp_NMS(hmps)
            hmp = filter_map[0, args.show_hmp_idx].cpu().numpy()
            plt.imshow(hmp)
            plt.show()

            dets = decoder.topK_channel(filter_map, K=50)

            dets = [det.cpu().numpy() for det in dets]
            # keymap是stride分辨率下的，没有缩放到image尺寸
            keymap = np.zeros((args.square_length,
                               args.square_length))
            for yy, xx in zip(dets[2][0, args.show_hmp_idx], dets[3][0, args.show_hmp_idx]):
                keymap[yy, xx] = 1
            plt.imshow(keymap)
            plt.show()

        if args.show_all_limbs:
            hmps = outputs[0][0][1]
            offs = outputs[1][0][1]
            sizeHW = (args.square_length, args.square_length)
            torch.cuda.synchronize()  # 需要吗？没有print之类的函数时，是需要的，等待cuda上的流程运行完
            t0 = time.time()
            hmps = torch.nn.functional.interpolate(hmps, size=sizeHW, mode="bicubic")
            offs = torch.nn.functional.interpolate(offs, size=sizeHW, mode="bicubic")
            torch.cuda.synchronize()
            t1 = time.time()
            tt1 = t1 - t0
            LOG.info('interpolation tims: %.6f', tt1)

            gen = decoder.LimbsCollect(1, 1, include_scale=True,
                                       topk=64, thre_hmp=args.thre_hmp)
            t2 = time.time()

            limbs = gen.generate_limbs(hmps, None, offs, 4 * torch.ones_like(hmps))
            limbs = limbs.cpu().numpy()

            torch.cuda.synchronize()  # 需要吗？
            tt2 = time.time() - t2
            LOG.info('keypoint pairing time: %.6f', tt2)
            t2 = time.time() - t0
            LOG.info('keypoint detection and pairing time: %.6f', t2)

            for ltype_i, connects in enumerate(limbs[0]):
                xyv1 = connects[:, 0:3]
                xyv2 = connects[:, 3:6]
                len_delta = connects[:, 8]
                for i in range(len(xyv1)):
                    if xyv1[i, 0] > 0 and xyv2[i, 0] > 0 and len_delta[i] <= args.dist_max:
                        x1, y1 = xyv1[i, :2].tolist()
                        x2, y2 = xyv2[i, :2].tolist()
                        plt.plot([x1, x2], [-y1, -y2], color='r')
                        plt.scatter([x1, x2], [-y1, -y2], color='g')
                        plt.xlim((0, args.square_length))
                        plt.ylim((-args.square_length, 0))
            plt.title('all candidate limbs')
            plt.show()

            assemble = decoder.GreedyGroup(args.person_thre, use_scale=True, sort_dim=2)
            t0 = time.time()
            worker_pool = multiprocessing.Pool(args.batch_size)
            # limbs_list = [(image_limb,) for i, image_limb in enumerate(limbs)]  # each element must be a tuple
            # starmap blocks the main process to wait all pools, while starmap_async is only little faster
            batch_poses = worker_pool.starmap(assemble.group_skeletons, zip(limbs))
            #
            image_poses = batch_poses[0]
            t1 = time.time() - t0
            LOG.info('\nGreedy grouping time: %.6f\n %d person poses', t1, len(image_poses))

            draw_color = ['g', 'r', 'b', 'c', 'm', 'y', 'k']
            for pose_idx, pose in enumerate(image_poses):
                color_id = pose_idx % len(draw_color)
                color = matplotlib.cm.get_cmap('tab20')((pose_idx % 20 + 0.05) / 20)
                xyvs = pose[:, :3]
                for i in range(len(xyvs)):
                    if xyvs[i, 0] > 0 and xyvs[i, 1] > 0 and xyvs[i, 2] > args.thre_hmp:
                        plt.scatter([xyvs[i, 0]], [-xyvs[i, 1]], color=color)
                        plt.xlim((0, args.square_length))
                        plt.ylim((-args.square_length, 0))
            plt.title('output of greedy assignment algorithm')
            plt.show()

        if isinstance(args.show_limb_idx, int):
            hmps = outputs[0][0][1].cpu().numpy()
            offs = outputs[1][0][1].cpu().numpy()
            offs[np.isinf(offs)] = 0
            hmp = hmps[0]
            off = offs[0]
            image = images.cpu().numpy()[0, ...].transpose((1, 2, 0))  # the first image

            skeleton = encoder.OffsetMaps.skeleton
            # first ,we should roughly rescale the image into the range of [0, 1]
            image = np.clip((image + 2.0) / 4.0, 0.0, 1.0)
            show.draw_limb_offset(hmp, image, off, args.show_limb_idx, skeleton, s=7, thre=0.2)

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


if __name__ == '__main__':
    log_level = logging.INFO  # logging.DEBUG
    # set RootLogger
    logging.basicConfig(
        level=log_level,
        # format="%(asctime)s 【 %(process)d 】 %(processName)s %(message)s"
    )

    main()
