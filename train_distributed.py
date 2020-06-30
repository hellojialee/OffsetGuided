"""Distributed training with Nvidia Apex"""
import argparse
import logging
import os
import sys
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


def train_cli():
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
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='freeze the pre-trained layers of the BaseNet, i.e. Backbone')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--warmup', action='store_true', default=False,
                        help='using warm-up learning rate')
    parser.add_argument('--checkpoint-path', '-p',
                        default='link2checkpoints_storage',
                        help='folder path checkpoint storage of the whole pose estimation model')
    parser.add_argument('--max-grad_norm', default=10, type=float,
                        help=(
                            "If the norm of the gradient vector exceeds this, "
                            "re-normalize it to have the norm equal to max_grad_norm"))

    group = parser.add_argument_group('apex configuration')
    group.add_argument("--local_rank", default=0, type=int)
    group.add_argument('--opt-level', type=str, default='O1')
    group.add_argument('--no-sync-bn', dest='sync_bn', action='store_false',
                       default=True,
                       help='enabling apex sync BN.')
    group.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    group.add_argument('--loss-scale', type=str, default=None)  # '1.0'
    group.add_argument('--channels-last', type=bool,
                       default=False)  # fixme: channel last may lead to 22% speed up
    group.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                       help='print frequency (default: 10)')

    group = parser.add_argument_group('optimizer configuration')
    group.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam'])
    group.add_argument('--learning-rate', type=float, default=2.5e-5,
                       metavar='LR',
                       help='learning rate')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                       help='momentum')
    group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                       metavar='W', help='weight decay (default: 1e-4)')

    args = parser.parse_args()

    if args.logging_output is None:
        args.logging_output = default_output_file(args)

    return args


def main():
    global best_loss, args
    best_loss = float('inf')
    start_epoch = 0
    args = train_cli()
    logs.configure(args)

    print(f"\nopt_level = {args.opt_level}")
    print(f"keep_batchnorm_fp32 = {args.keep_batchnorm_fp32}")
    print(f"loss_scale = {args.loss_scale}")
    print(f"CUDNN VERSION: {torch.backends.cudnn.version()}\n")

    if args.local_rank == 0:

        # build epoch recorder
        os.makedirs(args.checkpoint_path, exist_ok=True)
        recorder = open(os.path.join('./' + args.checkpoint_path, 'log'), 'a+')
        recorder.write('\ncmd line: {} \n \targs dict: {}'.format(' '.join(sys.argv), vars(args)))
        recorder.flush()
        recorder.close()

    torch.backends.cudnn.benchmark = True
    # print(vars(args))
    args.pin_memory = False
    if torch.cuda.is_available():
        args.pin_memory = True

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    # FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
    # the 'WORLD_SIZE' environment variable will also be set automatically.
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()  # get the current process id
        print("World Size is :", args.world_size)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    preprocess_transformations = [
        transforms.NormalizeAnnotations(),
        transforms.WarpAffineTransforms(args.square_length, aug_params=args,
                                        debug_show=False),
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

    if args.sync_bn:
        #  This should be done before model = DDP(model, delay_allreduce=True),
        #  because DDP needs to see the finalized model parameters
        import apex

        print("Using apex synced BN.")
        model = apex.parallel.convert_syncbn_model(model)

    # NOTICE! It should be called before constructing optimizer
    # if the module will live on GPU while being optimized.
    model.cuda()

    for param in model.parameters():
        if param.requires_grad:
            print('Parameters of network: Autograd')
            break

    # freeze the backbone layers
    if args.freeze:
        for name, param in model.named_parameters():
            if 'basenet' in name:
                param.requires_grad = False

    # optimizer = apex_optim.FusedSGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                                 lr=opt.learning_rate * args.world_size, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * args.world_size, momentum=args.momentum,
        weight_decay=args.weight_decay)
    # optimizer = apex_optim.FusedAdam(model.parameters(), lr=opt.learning_rate * args.world_size, weight_decay=1e-4)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)  # Dynamic loss scaling is used by default.
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    if args.resume:
        model, optimizer, start_epoch, best_loss = models.networks.load_model(
            model, args.checkpoint_whole, optimizer, resume_optimizer=True,
            drop_layers=False, optimizer2cuda=True)

    train_sampler = None
    val_sampler = None

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)

    # 创建数据加载器，在训练和验证步骤中喂数据
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.loader_workers,
                                               pin_memory=args.pin_memory,
                                               sampler=train_sampler,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.loader_workers,
                                             pin_memory=args.pin_memory,
                                             sampler=val_sampler,
                                             drop_last=True)

    # ############################# Train and Validate #############################
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(train_loader, train_sampler, model, lossfuns, optimizer, epoch)
        test(val_loader, val_sampler, model, lossfuns, optimizer, epoch)


def train(train_loader, train_sampler, model, criterion, optimizer, epoch):
    print('\n ############### Train phase, Epoch: {} #############'.format(
        epoch))
    torch.cuda.empty_cache()
    model.train()
    # disturb and allocation data differently at each epcoh
    # train_sampler make each GPU process see 1/(world_size) training samples per epoch
    if args.distributed:
        train_sampler.set_epoch(epoch)

    # adjust_learning_rate_cyclic(optimizer, epoch, start_epoch)  # start_epoch
    print(
        '\nLearning rate at this epoch is: %0.9f' % optimizer.param_groups[0][
            'lr'])  # scheduler.get_lr()[0]

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, (images, annos, metas) in enumerate(train_loader):
        # # ##############  Use fun of 'adjust learning rate' #####################
        adjust_learning_rate(optimizer, epoch, batch_idx, len(train_loader),
                             use_warmup=args.warmup)
        # print('\nLearning rate at this epoch is: %0.9f\n' % optimizer.param_groups[0]['lr'])  # scheduler.get_lr()[0]
        # # ##########################################################

        #  这允许异步 GPU 复制数据也就是说计算和数据传输可以同时进.
        images = images.cuda(non_blocking=True)
        anno_heads = [[x.cuda(non_blocking=True) for x in pack] for pack in
                      annos]

        optimizer.zero_grad()  # zero the gradient buff

        outputs = model(images)  # outputs of multiple headnets

        multi_losses = []
        for out, lossfun, anno in zip(outputs, criterion, anno_heads):
            multi_losses += list(lossfun(out, *anno))
        # weight the multi-task losses
        weighted_losses = [lam * l for lam, l in
                           zip(args.lambdas, multi_losses)]
        loss = sum(weighted_losses)  # args.lambdas defined in models.factory

        if loss.item() > 1e8:  # try to rescue the gradient explosion
            print("\nOh My God ! \nLoss is abnormal, drop this batch !")
            continue

        LOG.info({
            'type': f'train-at-rank{args.local_rank}',
            'epoch': epoch,
            'batch': batch_idx,
            'head_losses': [round(to_python_float(l.detach()) if torch.is_tensor(l) else l, 6)
                            for l in multi_losses],
            'loss': round(to_python_float(loss.detach()), 6),
        })

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)  # todo
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            # print不应该多用 会触发allreduce，而这个操作比较费时
            if args.distributed:
                # We manually reduce and average the metrics across processes. In-place reduce tensor.
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss),
                          images.size(0))  # update needs average and number
            torch.cuda.synchronize()  # 因为所有GPU操作是异步的，应等待当前设备上所有流中的所有核心完成，测试的时间才正确
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:  # Print them in the Process 0
                print('==================> Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f}) <================ \t'.format(
                    epoch, batch_idx, len(train_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses))

    # global best_loss
    # # DistributedSampler控制进入分布式环境的数据集以确保模型不是对同一个子数据集训练，以达到训练目标。
    #
    if args.local_rank == 0:
        # Write the log file each epoch.
        os.makedirs(args.checkpoint_path, exist_ok=True)
        recorder = open(os.path.join('./' + args.checkpoint_path, 'log'), 'a+')
        recorder.write('\nEpoch {}\ttrain_loss: {}'.format(epoch,
                                                           losses.avg))  # validation时不要\n换行
        recorder.flush()
        recorder.close()

        if losses.avg < float('inf'):  # < best_loss
            # Update the best_loss if the average loss drops
            best_loss = losses.avg  # todo: modify best_loss to best_AP

            save_path = './' + args.checkpoint_path + '/PoseNet_' + str(
                epoch) + '_epoch.pth'
            models.networks.save_model(save_path, epoch, best_loss, model, optimizer)


def test(val_loader, val_sampler, model, criterion, optimizer, epoch):
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
            weighted_losses = [lam * l for lam, l in
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

        if batch_idx % args.print_freq == 0:
            if args.distributed:
                # We manually reduce and average the metrics across processes. In-place reduce tensor.
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), images.size(0))  # update needs average and number
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:  # Print them in the Process 0
                print('==================> Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f}) <================ \t'.format(
                    epoch, batch_idx, len(val_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses))

    if args.local_rank == 0:  # Print them in the Process 0
        # Write the log file each epoch.
        os.makedirs(args.checkpoint_path, exist_ok=True)
        logger = open(os.path.join('./' + args.checkpoint_path, 'log'), 'a+')
        logger.write('\tval_loss: {}'.format(losses.avg))  # validation时不要\n换行
        logger.flush()
        logger.close()


def adjust_learning_rate(optimizer, epoch, step, len_epoch, use_warmup=False):
    factor = epoch // 15

    if epoch >= 78:
        factor = (epoch - 78) // 5

    lr = args.learning_rate * args.world_size * (0.2 ** factor)

    """Warmup"""
    if use_warmup:
        if epoch < 5:
            # print('=============>  Using warm-up learning rate....')
            lr = lr * float(1 + step + epoch * len_epoch) / (
                    5. * len_epoch)  # len_epoch=len(train_loader)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_cyclic(optimizer, current_epoch, start_epoch,
                                swa_freqent=5, lr_max=4e-5, lr_min=2e-5):
    epoch = current_epoch - start_epoch

    lr = lr_max - (lr_max - lr_min) / (swa_freqent - 1) * (
            epoch - epoch // swa_freqent * swa_freqent)
    lr = round(lr, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def reduce_tensor(tensor):
    # Reduces the tensor data on GPUs across all machines
    # If we print the tensor, we can get:
    # tensor(334.4330, device='cuda:1'), here is cuda:  cuda:1
    # tensor(340.1970, device='cuda:0'), here is cuda:  cuda:0
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


def default_output_file(args):
    out = 'logs/outputs/{}-{}'.format(args.basenet, '-'.join(args.headnets))
    if args.square_length != 512:
        out += '-edge{}'.format(args.square_length)
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out += '-{}.pkl'.format(now)

    return out


if __name__ == '__main__':
    main()
