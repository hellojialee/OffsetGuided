"""Simulation for the encoding and decoding process and check the performance on dataset"""
import os
import argparse
import logging
import cv2
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import math
import json
import torch
import torchvision
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import config
import data
import transforms
import models
import logs
from visualization import show
import encoder
import decoder
from utils.util import AverageMeter
from config.coco_data import (ANNOTATIONS_TRAIN, ANNOTATIONS_VAL, IMAGE_DIR_TRAIN, IMAGE_DIR_VAL,
                              ANNOTATIONS_TESTDEV, ANNOTATIONS_TEST, IMAGE_DIR_TEST, COCO_KEYPOINTS)

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

    parser.add_argument('--dataset', choices=('val', 'test', 'test-dev'), default='val',
                        help='dataset to evaluate')
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

    if args.dataset == 'val':
        args.image_dir = IMAGE_DIR_VAL
        args.annotation_file = ANNOTATIONS_VAL
    elif args.dataset == 'test':
        args.image_dir = IMAGE_DIR_TEST
        args.annotation_file = ANNOTATIONS_TEST
    elif args.dataset == 'test-dev':
        args.image_dir = IMAGE_DIR_TEST
        args.annotation_file = ANNOTATIONS_TESTDEV
    else:
        raise Exception

    if args.dataset in ('test', 'test-dev') and not args.all_images:
        print('force to use --all-images for this dataset because catIds are unknown')
        args.all_images = True
    return args


def run_images():
    count=0
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
    for batch_idx, (images, anno_heads, metas) in enumerate(val_loader):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)  # TODO: 更改成为CPU和CUDA兼容的程序

            anno_heads = [[x.cuda(non_blocking=True) for x in pack] for pack in
                          anno_heads]
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

        # if args.show_detected_poses:
        #     image_poses = batch_poses[0]
        #
        #     image_path = metas[0]['image_path']
        #     origi_img = cv2.imread(image_path)
        #     image = cv2.cvtColor(origi_img, cv2.COLOR_BGR2RGB)
        #     skeleton = processor.skeleton
        #     keypoint_painter = show.KeypointPainter(
        #         show_box=False,
        #         # color_connections=True, linewidth=5,
        #     )
        #     with show.image_canvas(image,
        #                            # output_path + '.keypoints.png',
        #                            show=True,
        #                            # fig_width=args.figure_width,
        #                            # dpi_factor=args.dpi_factor
        #                            ) as ax:
        #         keypoint_painter.keypoints(ax, image_poses[:, :, :3], skeleton=skeleton)

        if args.show_detected_poses:
            subset = batch_poses[0]

            image_path = metas[0]['image_path']
            canvas = cv2.imread(image_path)
            skeleton = [
                [12, 13],
                [13, 0],
                [13, 1],
                [6, 7],
                [0, 2],
                [2, 4],
                [1, 3],
                [3, 5],
                [0, 6],
                [1, 7],
                [6, 8],
                [8, 10],
                [7, 9],
                [9, 11]
            ]

            colors = [[128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255], [255, 0, 0], [255, 85, 0],
                      [255, 170, 0],
                      [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170],
                      [0, 255, 255],
                      [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                      [255, 0, 170],
                      [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255]]

            color_board = [0, 12, 9, 21, 13,14,10,11,18,15,19,20,16,17]
            color_idx = 0

            for i in range(len(skeleton)):  # 画出18个limb　Fixme：我设计了25个limb,画的limb顺序需要调整，相应color数也要增加
                for n in range(len(subset)):
                    index = subset[n][np.array(skeleton[i])][:, :3]
                    if 0 in index:  # 有-1说明没有对应的关节点与之相连,即有一个类型的part没有缺失，无法连接成limb
                        continue
                    # 在上一个cell中有　canvas = cv2.imread(test_image) # B,G,R order
                    cur_canvas = canvas.copy()
                    X = index.astype(int)[:, 1]
                    Y = index.astype(int)[:, 0]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 3), int(angle), 0,
                                               360, 1)

                    cv2.circle(cur_canvas, (int(Y[0]), int(X[0])), 4, color=[0, 0, 0], thickness=2)
                    cv2.circle(cur_canvas, (int(Y[1]), int(X[1])), 4, color=[0, 0, 0], thickness=2)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[color_board[color_idx]])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
                color_idx += 1

            # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_NORMAL 自动适合的窗口大小
            # cv2.imshow('result', canvas)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img_path = 'demo_imgs/' + str(count) + '_result.jpg'
            cv2.imwrite(img_path, canvas)
            count += 1

        if batch_idx % args.print_freq == 0:
            # to_python_float incurs a host<->device sync
            # torch.cuda.synchronize()
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


def validation(dump_name, args):
    annType = 'keypoints'
    prefix = 'person_keypoints'

    dataDir = 'data/link2COCO2017'
    dataset = 'val2017'

    annFile = args.annotation_file

    print(annFile)
    cocoGt = COCO(annFile)

    resFile = '%s/results/%s_%s_%s_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    print('the path of detected keypoint file is: ', resFile)
    os.makedirs(os.path.dirname(resFile), exist_ok=True)

    results_keypoints, validation_ids = run_images()

    json.dump(results_keypoints, open(resFile, 'w'))

    # ####################  COCO Evaluation ################
    sigmas = np.array([
        .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79,
        .79
    ]) / 10.0
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints_crowd', sigmas, use_area=False)
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
                                      args=args)  # 'val2017'
    print('\ntheoretical performance of our encoding and decoding')
