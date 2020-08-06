import logging
import re
import time
import argparse
import torch
import numpy as np
import multiprocessing
import decoder
from config.coco_data import (COCO_KEYPOINTS,
                              COCO_PERSON_SKELETON,
                              COCO_PERSON_WITH_REDUNDANT_SKELETON,
                              REDUNDANT_CONNECTIONS,
                              KINEMATIC_TREE_SKELETON)

LOG = logging.getLogger(__name__)


class PostProcess(object):
    def __init__(self, batch_size,
                 hmp_stride, off_stride,
                 keypoints, skeleton,
                 limb_collector, limb_grouper,
                 hmp_index=0, omp_index=1, feat_stage=-1):
        super(PostProcess, self).__init__()
        self.batch_size = batch_size
        self.hmp_stride = hmp_stride
        self.off_stride = off_stride
        self.keypoints = keypoints
        self.skeleton = skeleton
        self.limb_collect = limb_collector
        self.limb_group = limb_grouper
        self.hmp_index = hmp_index
        self.omp_index = omp_index
        self.feat_stage = feat_stage
        LOG.info('use the inferred feature maps at stage %d, '
                 'heatmap index is %d, offsetmap index is %d, ' 
                 'parallel execution of GreedyGroup in Pools with %d workers',
                 feat_stage, hmp_index, omp_index, batch_size)
        self.worker_pool = multiprocessing.Pool(batch_size)

    def __call__(self, features):
        # input feature maps regress by the network
        out_hmps, out_bghmp = features[self.hmp_index]

        out_offsets, out_spreads, out_scales = features[self.omp_index]

        # use the inferred heatmaps at the last stage/stack
        hmps = out_hmps[self.feat_stage]
        offs = out_offsets[self.feat_stage]
        scmps = out_scales[self.feat_stage]

        hmps = torch.nn.functional.interpolate(
            hmps, scale_factor=self.hmp_stride, mode="bicubic")
        offs = torch.nn.functional.interpolate(
            offs, scale_factor=self.off_stride, mode="bicubic")

        # convert torch.Tensor to numpy.ndarray
        limbs = self.limb_collect.generate_limbs(hmps, offs, scmps).cpu().numpy()
        # put grouping into Pools
        batch_poses = self.worker_pool.starmap(
            self.limb_group.group_skeletons, zip(limbs))

        return batch_poses


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def decoder_cli(parser):
    group = parser.add_argument_group('limb collections in post-processing')
    group.add_argument('--topk', default=48,
                       type=int,
                       help='select the top K responses on each heatmaps, '
                            'and hence leads to top K limbs of each type. '
                            'A bigger topk may not leads to better performance')
    group.add_argument('--thre-hmp', default=0.08,
                       type=float,
                       help='candidate kepoints below this response value '
                            'are moved outside the image boarder')
    group.add_argument('--min-len', default=3,
                       type=float,
                       help='length in pixels, clamp the candidate limbs of zero length to min_len')
    group.add_argument('--feat-stage', default=-1, type=int,
                       help='use the inferred feature maps at this stage to generate results')

    group = parser.add_argument_group('greedy grouping in post-processing')
    group.add_argument('--person-thre', default=0.08,
                       type=float,
                       help='threshold for pose instance scores.')
    group.add_argument('--sort-dim', default=2, choices=[2, 4],
                       type=int,
                       help='sort the person poses by the values at the this axis.'
                            '2th dim means keypoints score, 4th dim means limb score.')
    group.add_argument('--dist-max', default=10, type=float,
                       help='abandon limbs with delta offsets bigger than dist_max, '
                            'if keypoint scales are unavailable.')
    group.add_argument('--use-scale', default=True, type=boolean_string,
                       help='use the inferred keypoint scales as the criterion '
                            'to keep limbs (keypoint pairs)')


def parse_heads(head_name, stride):
    if head_name in ('hmp',
                     'hmps',
                     'heatmap',
                     'heatmaps') or \
            re.match('hmp[s]?([0-9]+)$', head_name) is not None:  # +: repeat one or more times
        # example [ho]: the first letter must be 'h' or 'o
        # [s]?: a single s exists or not,
        # [s]+: one or more s exist

        m = re.match('hmp[s]?([0-9]+)$', head_name)  # $ the end of string match
        if m is not None:
            n_keypoints = int(m.group(1))  # group eg: hmps17, hmps, 17
            assert n_keypoints == 17, f'{n_keypoints} keypoints not supported'

        else:
            n_keypoints = 17
            keypoints = COCO_KEYPOINTS
        return {'keypoints': keypoints, 'hmp_stride': stride}  # keypoint definition in dataset

    if head_name in ('omp',
                     'omps'
                     'offset',
                     'offsets') or \
            re.match('omp[s]?([0-9]+)$', head_name) is not None:
        if head_name in ('omp', 'omp19', 'omps', 'offset', 'offsets'):
            skeleton = COCO_PERSON_SKELETON  # default selection
        elif head_name in ('omp16',):
            skeleton = KINEMATIC_TREE_SKELETON
        elif head_name in ('omp29',):
            skeleton = COCO_PERSON_WITH_REDUNDANT_SKELETON
        elif head_name in ('omp6', 'omps6'):
            skeleton = REDUNDANT_CONNECTIONS
        else:
            raise Exception('unknown skeleton type of head')

        return {'skeleton': skeleton, 'omp_stride': stride}  # skeleton configuration of ours

    raise Exception('unknown head to create an encoder: {}'.format(head_name))


def decoder_factory(args):
    temp_dic = {}  # keypoins and  skeleton
    for hd_name, stride in zip(args.headnets, args.strides):
        temp_dic.update(parse_heads(hd_name, stride))

    limb_handler = decoder.LimbsCollect(temp_dic['hmp_stride'],
                                        temp_dic['omp_stride'],
                                        topk=args.topk,
                                        thre_hmp=args.thre_hmp,
                                        min_len=args.min_len,
                                        include_scale=args.include_scale,
                                        keypoints=temp_dic['keypoints'],
                                        skeleton=temp_dic['skeleton'])

    skeleton_grouper = decoder.GreedyGroup(args.person_thre,
                                           sort_dim=args.sort_dim,
                                           dist_max=args.dist_max,
                                           use_scale=args.use_scale,
                                           keypoints=temp_dic['keypoints'],
                                           skeleton=temp_dic['skeleton'])

    return PostProcess(args.batch_size,
                       temp_dic['hmp_stride'],
                       temp_dic['omp_stride'],
                       keypoints=temp_dic['keypoints'],
                       skeleton=temp_dic['skeleton'],
                       limb_collector=limb_handler,
                       limb_grouper=skeleton_grouper,
                       feat_stage=args.feat_stage)


def debug_parse_args():
    parser = argparse.ArgumentParser(description='Test decoder')
    # general
    parser.add_argument('--for-debug',
                        default=False,
                        action='store_true',
                        help='this parse is only for debug the code')

    decoder_cli(parser)
    args = parser.parse_args('--for-debug '.split())
    return args


if __name__ == '__main__':
    # for debug

    log_level = logging.DEBUG  # logging.INFO
    # set RootLogger
    logging.basicConfig(level=log_level)

    args = debug_parse_args()
    args.head_names = ['hmp', 'omp']
    args.strides = [4, 4, 4]
    args.batch_size = 8
    args.include_scale = False

    processor = decoder_factory(args)
    poses = processor.__call__()
    t = 2
