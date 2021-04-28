import logging
import re
import time
import argparse
import torch
import numpy as np
import multiprocessing
import decoder
from utils.util import boolean_string
import config
from config.coco_data import (COCO_KEYPOINTS,
                              COCO_PERSON_SKELETON,
                              COCO_PERSON_WITH_REDUNDANT_SKELETON,
                              DENSER_COCO_PERSON_SKELETON,
                              REDUNDANT_CONNECTIONS,
                              KINEMATIC_TREE_SKELETON)

LOG = logging.getLogger(__name__)


class PostProcess(torch.nn.Module):
    def __init__(self, batch_size,
                 hmp_stride, off_stride,
                 inter_mode,
                 keypoints, skeleton,
                 limb_collector, limb_grouper,
                 include_scale=False, include_jitter_offset=False,
                 hmp_index=0, omp_index=1, feat_stage=-1):
        super(PostProcess, self).__init__()
        self.batch_size = batch_size
        self.inter_mode = inter_mode  # type: str
        self.hmp_stride = hmp_stride
        self.off_stride = off_stride
        self.keypoints = keypoints
        self.skeleton = skeleton
        self.limb_collect = limb_collector
        self.limb_group = limb_grouper
        self.hmp_index = hmp_index
        self.omp_index = omp_index
        self.feat_stage = feat_stage
        self.include_scale = include_scale
        self.include_jitter_offset = include_jitter_offset
        self.keypoints_flips = config.heatmap_hflip(keypoints)
        self.limbs_flips = config.offset_hflip(keypoints, skeleton)
        LOG.info('use the inferred feature maps at stage %d, '
                 'heatmap index is %d, offsetmap index is %d, '
                 'interpolate the predicted heatmaps using %s, '
                 'parallel execution of GreedyGroup in Pools with %d workers',
                 feat_stage, hmp_index, omp_index, inter_mode, batch_size)
        self.worker_pool = multiprocessing.Pool(batch_size)

    def generate_poses(self, features, flip_test=False, cat_flip_offs=False, scored_off=False):
        # input feature maps regress by the network
        out_hmps, out_bghmp, out_jomps = features[self.hmp_index]  # type: torch.Tensor

        out_offsets, out_spreads, out_scales = features[self.omp_index]

        # use the inferred heatmaps at the last stage/stack
        # but model output is FP32
        hmps = out_hmps[self.feat_stage]
        jomps = out_jomps[self.feat_stage]
        offs = out_offsets[self.feat_stage]
        scmps = out_scales[self.feat_stage]

        vector_nd = 2
        # flip augmentation
        if flip_test:
            hmps, jomps, offs, scmps, vector_nd = self.flip_augment(hmps, jomps, offs, scmps, cat_flip_offs, vector_nd)

        if scored_off:
            joints_f, joints_t = decoder.offset.pack_jtypes(self.skeleton)
            offs = decoder.scored_offset(hmps, offs, joints_f, joints_t, kernel_size=3)

        hmps = torch.nn.functional.interpolate(  # todo: 可以只对hmps缩放，offs和scamps仍然在下采样分辨率下取
            hmps, scale_factor=self.hmp_stride, mode=self.inter_mode)
        # it's fine to resize offset only use 'bilinear'
        offs = torch.nn.functional.interpolate(
            offs, scale_factor=self.off_stride, mode='bilinear')

        if self.include_scale and isinstance(scmps, torch.Tensor):
            scmps = torch.nn.functional.interpolate(  # scales provide no increase to AP
                scmps, scale_factor=self.off_stride, mode=self.inter_mode)

        if self.include_jitter_offset and isinstance(jomps, torch.Tensor):
            # fixme: 在实际中，直接对jomps缩放插值是不合适的，可能通过heatmap找峰值点后在stride=4的jitter map上做
            #  refinement更加合理，避免插值带来的jitter offset失真！不确定，可以实际试一下；后处理全部都在stride=4上进行
            jomps = torch.nn.functional.interpolate(
                jomps, scale_factor=self.hmp_stride, mode='bilinear')

        # convert torch.Tensor to numpy.ndarray
        limbs = self.limb_collect.generate_limbs(hmps, jomps, offs, scmps, vector_nd).cpu().numpy()
        # put grouping into Pools
        batch_poses = self.worker_pool.starmap(
            self.limb_group.group_skeletons, zip(limbs))

        return batch_poses

    def flip_augment(self, hmps, jomps, offs, scmps, cat_flip_offs, vector_nd):
        n, limbsx2, h, w = offs.size()  # (2*N, 2*limbs, h, w)

        orig_hmps = hmps[:n // 2, ...]
        #  note: explicit index selection may be faster thant flip
        #  https://github.com/pytorch/pytorch/issues/229#issuecomment-579761958
        flip_hmps = torch.flip(hmps[n // 2:, ...], [-1])
        # hmps = torch.max(orig_hmps, flip_hmps[:, self.keypoints_flips, :, :])  # drop 2.6AP
        hmps = (orig_hmps + flip_hmps[:, self.keypoints_flips, :, :]) / 2

        # todo: have a check for flipping jomps, 直接翻转并x*(-1)然后平均吧
        if self.include_jitter_offset and isinstance(jomps, torch.Tensor):
            orgi_jomps = jomps[:n // 2, ...]  # (N, 2, h, w)
            flip_jomps = torch.flip(jomps[n // 2:, ...], [-1])
            flip_jomps[:, ::2, :, :] *= -1
            jomps = (orgi_jomps + flip_jomps) / 2

        if cat_flip_offs:
            LOG.info('concatenate the flipped offsets along the original offsets')
            # offset flip merge of increasing to 4D vector space, drop 0.5AP
            offs = offs.view((n, -1, 2, h, w))  # (2*N, limbs, 2, h, w)
            orig_offs = offs[:n // 2, ...]  # (N, limbs, 2, h, w)
            reserve_offs = offs[:n // 2, self.limbs_flips[1], ...].clone()  # (N, uniques, 2, h, w)
            flip_offs = torch.flip(offs[n // 2:, ...], [-1])
            # flip the offset_x orientation
            flip_offs[:, :, ::2, :, :] *= -1.0  # (N, limbs, 2, h, w)
            offs = torch.cat((orig_offs, flip_offs[:, self.limbs_flips[0], ...]), dim=2)  # (2*N, limbs, 2, h, w)
            offs[:, self.limbs_flips[1], 2:, :] = reserve_offs
            offs = offs.view((n, -1, h, w))  # # (2N, limbs, 4, h, w) -> (2N, 4*limbs, h, w)
            vector_nd = 4

        else:
            # offset flip merge of vector addition
            offs = offs.view((n, -1, 2, h, w))  # (2*N, limbs, 2, h, w)
            orig_offs = offs[:n // 2, ...]  # (N, limbs, 2, h, w)
            reserve_offs = offs[:n // 2, self.limbs_flips[1], ...].clone()
            flip_offs = torch.flip(offs[n // 2:, ...], [-1])
            # flip the offset_x orientation
            flip_offs[:, :, ::2, :, :] *= -1.0  # (N, limbs, 2, h, w)
            offs = (orig_offs + flip_offs[:, self.limbs_flips[0], ...]) / 2
            offs[:, self.limbs_flips[1], ...] = reserve_offs
            offs = offs.view((n//2, -1, h, w))  # (N, 2*limbs, h, w)

        if self.include_scale and isinstance(scmps, torch.Tensor):
            orig_scmps = scmps[:n // 2, ...]
            flip_scmps = torch.flip(scmps[n // 2:], [-1])
            scmps = (orig_scmps + flip_scmps[:, self.keypoints_flips, :, :]) / 2

        return hmps, jomps, offs, scmps, vector_nd


def decoder_cli(parser):
    group = parser.add_argument_group('limb collections in post-processing')
    group.add_argument('--resize-mode', default="bicubic", choices=["bilinear", "bicubic"],
                       type=str,
                       help='interpolation mode for resizing the keypoint heatmaps.')
    group.add_argument('--topk', default=48,
                       type=int,
                       help='select the top K responses on each heatmaps, '
                            'and hence leads to top K limbs of each type. '
                            'A bigger topk may not leads to better performance')
    group.add_argument('--thre-hmp', default=0.06,
                       type=float,
                       help='candidate kepoints below this response value '
                            'are moved outside the image boarder')
    group.add_argument('--min-len', default=0.5,
                       type=float,
                       help='length in pixels, clamp the candidate limbs of zero length to min_len')
    group.add_argument('--feat-stage', default=-1, type=int,
                       help='use the inferred feature maps at this stage to generate results')

    group = parser.add_argument_group('greedy grouping in post-processing')
    group.add_argument('--person-thre', default=0.06,
                       type=float,
                       help='threshold for pose instance scores, '
                            'but COCO evaluates the top k instances')
    group.add_argument('--sort-dim', default=2, choices=[2, 4],
                       type=int,
                       help='sort the person poses by the values at the this axis.'
                            ' 2th dim means keypoints score, 4th dim means limb score.')
    group.add_argument('--dist-max', default=20, type=float,
                       help='abandon limbs with delta offsets bigger than dist_max, '
                            'only useful when keypoint scales are not used because use-scale'
                            'will overlap the smaller dist-max')
    group.add_argument('--use-scale', default=True, type=boolean_string,
                       help='only effective when we set --include-scale in the network'
                            'use the inferred keypoint scales as the criterion '
                            'to keep limbs (keypoint pairs)')
    group.add_argument('--use-jitter-offset', default=True, type=boolean_string,
                       help='only effective when we set --include-jitter-offset in the network'
                            'use the inferred jitter offset to refine the precision drop of keypoint localization')


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
        elif head_name in ('omp31',):
            skeleton = COCO_PERSON_WITH_REDUNDANT_SKELETON
        elif head_name in ('omp44',):
            skeleton = DENSER_COCO_PERSON_SKELETON
        elif head_name in ('omp25', 'omps25'):
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
                                        include_jitter_offset=args.include_jitter_offset,
                                        include_scale=args.include_scale,
                                        use_jitter_offset=args.use_jitter_offset,
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
                       args.resize_mode,
                       keypoints=temp_dic['keypoints'],
                       skeleton=temp_dic['skeleton'],
                       limb_collector=limb_handler,
                       limb_grouper=skeleton_grouper,
                       include_scale=args.include_scale,
                       include_jitter_offset=args.include_jitter_offset,
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
    poses = processor.generate_poses()
    t = 2
