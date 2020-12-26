"""Simulation for the encoding and decoding process and check the performance on dataset"""
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
import config
import data
import transforms
import models
import logs
from visualization import show
import encoder
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


def evaluate_cli():
    parser = argparse.ArgumentParser(
        # __doc__: current module's annotation (or module.a_function's annotation)
        description=__doc__,
        # --help text with default values, e.g., gaussian threshold (default: 0.1)
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    logs.cli(parser)
    data.data_cli(parser)
    encoder.encoder_cli(parser)
    models.net_cli(parser)
    decoder.decoder_cli(parser)

    parser.add_argument('--resume', '-r', action='store_true', default=False,
                        help='resume from checkpoint')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--checkpoint-path', '-p',
                        default='link2checkpoints_storage',
                        help='folder path checkpoint storage of the whole pose estimation model')
    parser.add_argument('--show-detected-poses', action='store_true', default=False,
                        help='show the final results')
    parser.add_argument('--long-edge', default=640, type=int,
                        help='long edge of input images')

    group = parser.add_argument_group('apex configuration')
    group.add_argument("--local_rank", default=0, type=int)
    group.add_argument('--opt-level', type=str, default='O2')
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


def run_images():
    result_keypoints = []
    result_image_ids = []

    print(f"\nopt_level = {args.opt_level}")
    print(f"keep_batchnorm_fp32 = {args.keep_batchnorm_fp32}")
    print(f"CUDNN VERSION: {torch.backends.cudnn.version()}\n")

    torch.backends.cudnn.benchmark = True
    # print(vars(args))
    args.pin_memory = False
    if torch.cuda.is_available():
        args.pin_memory = True

    args.world_size = 1
    args.square_length = args.long_edge

    preprocess_transformations = [
        transforms.NormalizeAnnotations(),
        # transforms.RandomApply(transforms.AnnotationJitter(shift=1), 0.4),
        transforms.RescaleLongAbsolute(args.long_edge),
        transforms.CenterPad(args.long_edge),
        # transforms.RightDownPad(args.long_edge)  # CenterPad leads to higher metrics
    ]
    # ##### WarpAffineTransforms #######
    # preprocess_transformations = [
    #     transforms.NormalizeAnnotations(),
    #     transforms.WarpAffineTransforms(args.square_length, crop_roi=False,
    #                                     aug_params=transforms.FixedAugParams()), ]

    preprocess_transformations += [
        transforms.ImageTransform(torchvision.transforms.ToTensor()),
        transforms.ImageTransform(
            torchvision.transforms.Normalize(mean=config.data_mean,
                                             std=config.data_std)),
    ]
    preprocess = transforms.Compose(preprocess_transformations)
    target_transform = encoder.encoder_factory(args, args.strides)

    train_loader, val_loader = data.dataloader_factory(args, preprocess, target_transform)

    processor = decoder.decoder_factory(args)

    batch_time = AverageMeter()
    end = time.time()
    for batch_idx, (images, annos, metas) in enumerate(val_loader):

        images = images.cuda(non_blocking=True)

        anno_heads = [[x.cuda(non_blocking=True) for x in pack] for pack in
                      annos]
        # feed the ground-truth labels into the decoder, i.e., processor.generate_poses
        features = [[[anno_heads[0][0]], [None], [anno_heads[0][2]]], [[anno_heads[1][0]], [None], [None]]]
        # post-processing for generating individual poses
        batch_poses = processor.generate_poses(features)

        # #########################################################################
        # ############# inverse the keypoint into the original image space ############
        # #########################################################################
        for index, (image_poses, image_meta) in enumerate(zip(batch_poses, metas)):
            # you can change to affine_keypoint_inverse or annotation_inverse
            subset = preprocess.annotations_inverse(image_poses, image_meta)
            batch_poses[index] = subset

            image_id = image_meta['image_id']

            # #########################################################################
            # ############# collect the detected person poses and image ids ###########
            # #########################################################################
            result_image_ids.append(image_id)
            # last dim of subset: [x, y, v, s, limb_score, ind]
            subset[:, :, :2] = np.around(subset[:, :, :2], 2)
            # print('detection \n', subset.astype(int))

            for i, person in enumerate(subset.astype(float)):  # json cannot write float32
                keypoints_list = []
                v = []
                for xyv in person[:, :3]:
                    v.append(xyv[2])
                    keypoints_list += [xyv[0], xyv[1],
                                       1 if xyv[0] > 0 and xyv[1] > 0 else 0]

                result_keypoints.append({
                    'image_id': image_id,
                    'category_id': 1,  # person category
                    'keypoints': keypoints_list,
                    'score': sum(v) / len(v),  # todo: person pose scare
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

    return result_keypoints, result_image_ids


def validation(dump_name, dataset='val2017'):
    annType = 'keypoints'
    prefix = 'person_keypoints'

    dataDir = 'data/link2COCO2017'

    # # # #############################################################################
    # For evaluation on validation set
    if dataset == 'val2017':
        annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataset)

    # #############################################################################
    # For evaluation on test-dev set
    elif dataset == 'test2017':
        annFile = 'data/dataset/coco/link2coco2017/annotations_trainval_info/image_info_test-dev2017.json'  # image_info_test2017.json
    else:
        raise Exception('unknown dataset')

    print(annFile)
    cocoGt = COCO(annFile)

    resFile = '%s/results/%s_%s_%s_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    print('the path of detected keypoint file is: ', resFile)
    os.makedirs(os.path.dirname(resFile), exist_ok=True)

    results_keypoints, validation_ids = run_images()

    json.dump(results_keypoints, open(resFile, 'w'))

    # ####################  COCO Evaluation ################
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = validation_ids  # only part of the person images are evaluated
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
    args = evaluate_cli()

    eval_result_original = validation(dump_name='hourglass104_focal_epoch_70_640_input_1scale',
                                      dataset='val2017')  # 'val2017'
    print('\ntheoretical performance of our encoding and decoding')
