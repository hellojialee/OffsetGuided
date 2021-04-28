"""Evaluate the one-scale performance on MSCOCO dataset"""
import os
import argparse
import logging
import cv2
import matplotlib.pyplot as plt
import time
import warnings
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

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # choose the available GPUs

LOG = logging.getLogger(__name__)

ANNOTATIONS_TRAIN = 'data/link2COCO2017/annotations/person_keypoints_train2017.json'
ANNOTATIONS_VAL = 'data/link2COCO2017/annotations/person_keypoints_val2017.json'
IMAGE_DIR_TRAIN = 'data/link2COCO2017/train2017'
IMAGE_DIR_VAL = 'data/link2COCO2017/val2017'

ANNOTATIONS_TESTDEV = 'data/link2COCO2017/annotations_trainval_info/image_info_test-dev2017.json'
ANNOTATIONS_TEST = 'data/link2COCO2017/annotations_trainval_info/image_info_test2017.json'
IMAGE_DIR_TEST = 'data/link2COCO2017/test2017/'


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

    parser.add_argument('--dataset', choices=('val', 'test', 'test-dev'), default='val',
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
                    'keypoints': np.zeros((17 * 3,)).tolist(),
                    'score': 0.01,
                })

        if args.show_detected_poses:
            image_poses = batch_poses[0]

            image_path = metas[0]['image_path']
            orig_img = cv2.imread(image_path)
            rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            skeleton = processor.skeleton
            keypoint_painter = show.KeypointPainter(
                show_box=False,
                # color_connections=True, linewidth=5,
            )
            with show.image_canvas(rgb_img,
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
                0, batch_idx, len(data_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time))

    return result_keypoints, result_image_ids


def validation(annFile, dump_name, dataset):
    prefix = 'person_keypoints'

    dataDir = 'data/link2COCO2017'

    print(annFile)
    cocoGt = COCO(annFile)

    resFile = '%s/results/%s_%s_%s_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    print('======================>')
    print('The path of detected keypoint file is: ', resFile)
    os.makedirs(os.path.dirname(resFile), exist_ok=True)

    results_keypoints, validation_ids = run_images()

    json.dump(results_keypoints, open(resFile, 'w'))

    # ####################  COCO Evaluation ################
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='keypoints')
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


