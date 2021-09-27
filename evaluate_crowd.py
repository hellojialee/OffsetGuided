"""Evaluate the one-scale performance on CrowdPose dataset"""
import os
import argparse
import logging
import cv2
import matplotlib.pyplot as plt
import time
import warnings
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
import decoder
from utils.util import AverageMeter

from config.coco_data import (ANNOTATIONS_TRAIN, ANNOTATIONS_VAL, IMAGE_DIR_TRAIN, IMAGE_DIR_VAL,
                              ANNOTATIONS_TESTDEV, ANNOTATIONS_TEST, IMAGE_DIR_TEST, COCO_KEYPOINTS)

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    import apex.optimizers as apex_optim
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example.")

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # choose the available GPUs

LOG = logging.getLogger(__name__)


def evaluate_cli():
    parser = argparse.ArgumentParser(
        # __doc__: current module's annotation (or module.a_function's annotation)
        description=__doc__,
        # --help text with default values, e.g., gaussian threshold (default: 0.1)
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    logs.cli(parser)
    models.net_cli(parser)
    decoder.decoder_cli(parser)

    parser.add_argument('--dump-name',  # TODO: edit this name each evaluation
                        default='hourglass104_focal_l2_instance_l1_sqrt_epoch_77__distmax40_640_input_1scale_flip_hmpoff_gamma2_thre004',
                        type=str, help='detection file name')

    parser.add_argument('--dataset', choices=('val', 'test', 'test-dev', 'train'), default='val',
                        help='dataset to evaluate')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--long-edge', default=640, type=int,
                        help='long edge of input images')
    parser.add_argument('--fixed-height', action='store_true', default=False,
                        help='resize input images to the fixed height of long_edge')
    parser.add_argument('--flip-test', action='store_true', default=False,
                        help='flip augmentation during testing')
    parser.add_argument('--cat-flip-offset', action='store_true', default=False,
                        help='offset flip merge of increasing to 4D vector space')
    parser.add_argument('--loader-workers', default=8, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--all-images', default=False, action='store_true',
                        help='run over all images irrespective of catIds')

    parser.add_argument('--resume', '-r', action='store_true', default=False,
                        help='resume from checkpoint')
    parser.add_argument('--checkpoint-path', '-p',
                        default='link2checkpoints_storage',
                        help='folder path checkpoint storage of the whole pose estimation model')
    parser.add_argument('--show-detected-poses', action='store_true', default=False,
                        help='show the final results')

    group = parser.add_argument_group('apex configuration')
    group.add_argument("--local_rank", default=0, type=int)
    # full precision O0  # mixture precision O1 # half precision O2
    group.add_argument('--opt-level', type=str, default='O2')
    group.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    group.add_argument('--loss-scale', type=str, default=None)  # '1.0'
    # todo: channel last may lead to 22% speed up however not obvious here
    #  channels Last memory format is implemented for 4D NCWH Tensors only.
    group.add_argument('--channels-last', default=False, action='store_true',
                       help='channel last may lead to 22% speed up')
    group.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                       help='print frequency (default: 10)')

    # args = parser.parse_args(
    #     '--checkpoint-whole link2checkpoints_storage/PoseNet_18_epoch.pth --resume --no-pretrain'.split())
    args = parser.parse_args()

    if args.dataset == 'val':
        args.image_dir = IMAGE_DIR_VAL
        args.annotation_file = ANNOTATIONS_VAL
    elif args.dataset == 'train':
        args.image_dir = IMAGE_DIR_TRAIN
        args.annotation_file = ANNOTATIONS_TRAIN
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
    count = 0
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

    if args.fixed_height:
        try:
            assert args.batch_size == 1
        except AssertionError:
            warnings.warn(f'forcibly fix the height of the input rgb_img to '
                          f'{args.long_edge: d} and set batch_size=1"')
            args.batch_size = 1

        preprocess_transformations = [
            transforms.NormalizeAnnotations(),
            # transforms.RescaleRelative(scale_factor=1),  # original input
            transforms.RescaleHighAbsolute(args.long_edge),
            transforms.RightDownPad(args.max_stride),
        ]
    else:
        LOG.info('you resize the longer edge of the input rgb_img to %d ', args.long_edge)
        preprocess_transformations = [
            transforms.NormalizeAnnotations(),
            transforms.RescaleLongAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
        ]

    preprocess_transformations += [
        transforms.ImageTransform(torchvision.transforms.ToTensor()),
        transforms.ImageTransform(
            torchvision.transforms.Normalize(mean=config.data_mean,
                                             std=config.data_std)),
    ]
    preprocess = transforms.Compose(preprocess_transformations)

    dataset = data.CocoKeypoints(
        args.image_dir,
        annFile=args.annotation_file,
        preprocess=preprocess,
        all_persons=True,  # we do not know if people are exitst in each rgb_img of test-dev
        all_images=args.all_images,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        collate_fn=data.collate_images_anns_meta)

    model, _ = models.model_factory(args)

    model.cuda()  # .to(memory_format=torch.channels_last)

    if args.resume:
        model, _, start_epoch, best_loss, _ = models.networks.load_model(
            model, args.checkpoint_whole, optimizer=None, resume_optimizer=False,
            drop_layers=False, load_amp=False)

    processor = decoder.decoder_factory(args)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interpretation with argparse.
    # make processor as Module (rewrite forward method) and wrap it 测试后发现没有明显加速效果
    [model] = amp.initialize([model],  # , processor
                             opt_level=args.opt_level,
                             keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                             loss_scale=args.loss_scale)

    model.eval()

    batch_time = AverageMeter()
    end = time.time()  # Notice: CenterNet 和 pifpaf 应该没有包含数据加载的时间，所以确认一下这个计时，考虑换个位置
    for batch_idx, (images, annos, metas) in enumerate(data_loader):

        images = images.cuda(non_blocking=True)  # .to(memory_format=memory_format)

        if args.flip_test:
            images = torch.cat((images, torch.flip(images, [-1])))  # images tensor shape = (N, 3, H, W)

        with torch.no_grad():
            outputs = model(images)  # outputs of multiple headnets

        # post-processing for generating individual poses
        batch_poses = processor.generate_poses(
            outputs,
            flip_test=args.flip_test,
            cat_flip_offs=args.cat_flip_offset
        )

        # #########################################################################
        # ############# inverse the keypoint into the original rgb_img space ############
        # #########################################################################
        for index, (image_poses, image_meta) in enumerate(zip(batch_poses, metas)):
            subset = preprocess.annotations_inverse(image_poses, image_meta)
            batch_poses[index] = subset  # numpy array [M * [ 17 * [x, y, v, s, limb_score, ind]]]

            image_id = image_meta['image_id']

            # #########################################################################
            # ############# collect the detected person poses and rgb_img ids ###########
            # #########################################################################
            result_image_ids.append(image_id)
            # last dim of subset: [x, y, v, s, limb_score, ind]
            subset[:, :, :2] = np.around(subset[:, :, :2], 2)
            for i, person in enumerate(subset.astype(float)):
                keypoints_list = []
                v = []
                for xyv in person[:, :3]:
                    v.append(xyv[2])
                    keypoints_list += [
                        xyv[0], xyv[1], 1 if xyv[0] > 0 or xyv[1] > 0 else 0
                    ]
                result_keypoints.append({
                    'image_id': image_id,
                    'category_id': 1,  # person category
                    'keypoints': keypoints_list,
                    'score': sum(v) / len(v),
                    #  the criterion of person pose score
                    #  we actually did at decoder.group.GreedyGroup._delete_sort
                })

            # force at least one annotation per rgb_img following PIFPAF
            # but make no difference to the result
            if not len(subset):
                result_keypoints.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': np.zeros((len(COCO_KEYPOINTS) * 3,)).tolist(),
                    'score': 0.01,
                })

        # if args.show_detected_poses:
        #     image_poses = batch_poses[0]
        #
        #     image_path = metas[0]['image_path']
        #     orig_img = cv2.imread(image_path)
        #     rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        #     skeleton = processor.skeleton
        #     keypoint_painter = show.KeypointPainter(
        #         show_box=False,
        #         # color_connections=True, linewidth=5,
        #     )
        #     with show.image_canvas(rgb_img,
        #                             # 'hehkeypoints.png',  # fixme, comment this line
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

            color_board = [0, 12, 9, 21, 13, 14, 10, 11, 18, 15, 19, 20, 16, 17]
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
            img_path = 'demo_imgs/'+str(count)+'_result.jpg'
            cv2.imwrite(img_path, canvas)
            count += 1

        if batch_idx % args.print_freq == 0:
            # to_python_float incurs a host<->device sync
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            print('==================> Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f}) <================ \t'.format(
                0, batch_idx, len(data_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time))

    return result_keypoints, result_image_ids


def validation(annFile, dump_name, dataset):
    prefix = 'person_keypoints'

    dataDir = 'data/link2CrowdPose'

    print('path to the ground truth annotation file is:', annFile)
    cocoGt = COCO(annFile)

    resFile = '%s/results/%s_%s_%s_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    print('======================>')
    print('The path of detected keypoint file is: ', resFile)
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

    print(args.dump_name)   # todo: 添加 NN inference 和 decoding time记录
    eval_result_original = validation(args.annotation_file,
                                      dump_name=args.dump_name,
                                      dataset=args.dataset)
    print('\nEvaluation finished!')


