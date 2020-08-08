"""Evaluate the one-scale performance on MSCOCO dataset"""
import os
import argparse
import logging
import cv2
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import json
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import data
import transforms
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
    parser.add_argument('--long-edge', default=640, type=int,
                        help='long edge of input images')

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


def run_images(resFile):
    print(f"\nopt_level = {args.opt_level}")
    print(f"keep_batchnorm_fp32 = {args.keep_batchnorm_fp32}")
    print(f"CUDNN VERSION: {torch.backends.cudnn.version()}\n")

    torch.backends.cudnn.benchmark = True
    # print(vars(args))
    args.pin_memory = False
    if torch.cuda.is_available():
        args.pin_memory = True

    args.world_size = 1

    if args.batch_size == 1:
        preprocess_transformations = [
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.RightDownPad(args.max_stride),
            transforms.RandomApply(transforms.AnnotationJitter(), 0),
        ]
    else:
        preprocess_transformations = [
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
            transforms.RandomApply(transforms.AnnotationJitter(), 0),
        ]

    preprocess_transformations += [
        # transforms.RandomApply(transforms.JpegCompression(), 0.1),
        transforms.RandomApply(transforms.ColorTint(), 0),
        transforms.ImageTransform(torchvision.transforms.ToTensor()),
        transforms.ImageTransform(
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])),
    ]
    preprocess = transforms.Compose(preprocess_transformations)

    train_loader, val_loader = data.dataloader_factory(args, preprocess)

    model, _ = models.model_factory(args)

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

    model.eval()

    results_keypoints = []

    batch_time = AverageMeter()
    end = time.time()
    for batch_idx, (images, annos, metas) in enumerate(val_loader):

        images = images.cuda(non_blocking=True)

        with torch.no_grad():
            outputs = model(images)  # outputs of multiple headnets

        # post-processing for generating individual poses
        batch_poses = processor.generate_poses(outputs)

        # #########################################################################
        # ############# inverse the keypoint into the original image space ############
        # #########################################################################
        for index, (image_poses, image_meta) in enumerate(zip(batch_poses, metas)):
            subset = preprocess.annotations_inverse(image_poses, image_meta)
            batch_poses[index] = subset

            image_id = image_meta['image_id']

            for i, person in enumerate(subset.astype(float)):  # last dim of subset: [x, y, v, s, limb_score, ind]
                keypoints_list = []
                v = []
                for xyv in person[:, :3]:
                    v.append(xyv[2])
                    if xyv[2] > 0:
                        keypoints_list += [xyv[0], xyv[1], 1]
                    else:
                        keypoints_list += [0, 0, 1]

                results_keypoints.append({
                    'image_id': image_id,
                    'category_id': 1,  # person category
                    'keypoints': keypoints_list,
                    'score': sum(v) / len(v),
                })

        if args.show_detected_poses:
            image_poses = batch_poses[0]

            image_path = metas[0]['image_path']
            origi_img = cv2.imread(image_path)
            image = cv2.cvtColor(origi_img, cv2.COLOR_BGR2RGB)
            skeleton = processor.skeleton
            keypoint_painter = show.KeypointPainter(
                show_box=False,
                # color_connections=True, linewidth=5,
            )
            with show.image_canvas(image,
                                   # output_path + '.keypoints.png',
                                   show=True,
                                   # fig_width=args.figure_width,
                                   # dpi_factor=args.dpi_factor
                                   ) as ax:
                keypoint_painter.keypoints(ax, image_poses[:, :, :3], skeleton=skeleton)

        if batch_idx % args.print_freq == 0:
            # to_python_float incurs a host<->device sync
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            print('==================> Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f}) <================ \t'.format(
                0, batch_idx, len(val_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time))

    json.dump(results_keypoints, open(resFile, 'w'))


def validation(dump_name, validation_ids=None, dataset='val2017'):
    annType = 'keypoints'
    prefix = 'person_keypoints'

    dataDir = 'data/link2COCO2017'

    # # # #############################################################################
    # For evaluation on validation set
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataset)
    print(annFile)
    cocoGt = COCO(annFile)

    if validation_ids == None:
        validation_ids = cocoGt.getImgIds(catIds=cocoGt.getCatIds(catNms=['person']))[:args.n_images_val]
    # # #############################################################################

    # #############################################################################
    # For evaluation on test-dev set
    # annFile = 'data/dataset/coco/link2coco2017/annotations_trainval_info/image_info_test-dev2017.json' # image_info_test2017.json
    # cocoGt = COCO(annFile)
    # validation_ids = cocoGt.getImgIds()
    # #############################################################################

    resFile = '%s/results/%s_%s_%s_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    print('the path of detected keypoint file is: ', resFile)
    os.makedirs(os.path.dirname(resFile), exist_ok=True)

    run_images(resFile)

    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = validation_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval


if __name__ == '__main__':
    log_level = logging.INFO  # logging.DEBUG
    # set RootLogger
    logging.basicConfig(
        level=log_level,
    )
    global best_loss, args
    best_loss = float('inf')
    args = demo_cli()

    eval_result_original = validation(dump_name='hourglass104_focal_epoch_70_640_input_1scale',
                                      dataset='val2017')  # 'val2017'
    print('over!')
