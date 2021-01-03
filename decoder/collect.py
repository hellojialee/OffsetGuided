"""Generate and collect all candidate keypoint pairs"""
import logging
import time
import torch
import numpy as np
from decoder import joint_dets
from config.coco_data import (COCO_KEYPOINTS,
                              COCO_PERSON_SKELETON,
                              COCO_PERSON_WITH_REDUNDANT_SKELETON,
                              REDUNDANT_CONNECTIONS,
                              KINEMATIC_TREE_SKELETON)

LOG = logging.getLogger(__name__)


class LimbsCollect(object):
    """
    Collect all **candidate** keypoints and pair them into limbs
    on the basis of guiding offset vectors.

    ***If multiple offsetmap Tensors are generated by network, you could concatenate
      them along the dim=2 and get a bigger Tensor (N, Q1+Q2+Q3, H, W).***

    Drawback: can not deal with isolated keypoints well.

    Attributes:
        hmp_s (int): stride of coordinate Unit of heatmaps with respect to that of input image
        off_s (int): stride of coordinate Unit of offetmaps with respect to that of input image
        keypoints (list): keypoints in human skeleton
        topk (int): select the top K responses on each heatmaps, and hence leads to top K limbs of each type
        min_len (int): length in pixels, in case of the zero length of limb
        include_scale (bool): use the inferred keypoint scales.
        skeleton (list): limb sequence, i.e., keypoint connections in the human skeleton
        thre_hmp (float): candidate kepoints below this value are moved outside the image boarder
    """

    def __init__(self, hmp_s, off_s, *, topk=40, thre_hmp=0.08, min_len=3,
                 include_jitter_offset=False, include_scale=False, use_jitter_offset=True,
                 keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON):
        LOG.info('number of skeleton limbs: %d, '  # separate the long string without comma
                 'response threshold to drop keypoints: threshold=%.4f',
                 len(skeleton), thre_hmp)
        self.hmp_s = hmp_s
        self.off_s = off_s
        self.resize_factor = off_s / hmp_s
        LOG.info('hmp stride: %d, off stride: %d \n '
                 'unify the heatmap coordinate unit and offset coordinate '
                 'unit using rescale factor: %.3f',
                 hmp_s, off_s, self.resize_factor)
        self.keypoints = keypoints
        self.skeleton = skeleton
        self.K = topk
        self.thre_hmp = thre_hmp
        self.min_len = min_len
        self.include_jitter_offset = include_jitter_offset
        LOG.info('inferred keypoint jitter offsets are available: %s', include_jitter_offset)
        self.include_scale = include_scale
        LOG.info('inferred kepoint scales are available: %s', include_scale)
        self.use_jitter_offset = use_jitter_offset
        self.jtypes_f, self.jtypes_t = self.pack_jtypes(skeleton)

    def generate_limbs(self,
                       hmps_hr: torch.Tensor,
                       jomps_hr: torch.Tensor,
                       offs_hr: torch.Tensor,
                       scmps_hr,
                       vector_nd=2) -> torch.Tensor:
        """
        Generate all candidate limbs between adjacent keypoints in the human skeleton tree,
         on the basis of the regressed heatmaps and offsetmaps

        Args:
            hmps_hr (Tensor): with shape (N, C, H, W)
            jomps_hr (Tensor): jitter offset to nearest keypoints with shape (N, 2, H, W)
            offs_hr (Tensor): with shape (N, L*2, H, W)
            scmps_hr (Tensor): (N, C, H, W), feature map of inferred keypoint scales.
            vector_nd (int): dimension number of each offset vector

        Returns: a Tensor containing all candidate limbs information in a batch images (batch=N here)
        """
        assert hmps_hr.shape[-2:] == offs_hr.shape[-2:], 'spatial resolution should be equal'

        LOG.debug('input size of heatmaps: %d * %d, '
                  'input size of offmaps: %d * %d ',
                  hmps_hr.shape[2], hmps_hr.shape[3],
                  offs_hr.shape[2], offs_hr.shape[3])
        start_time = time.time()  # fixme: 添加cuda 因异步运行的造成的统计时间问题处理
        # ########################################################################
        # ################ get all information for all limb endpoints ############
        # ########################################################################
        # shape of each item of dets: (N, 17, K) or may be (N, C, K),
        # in which K equals self.K and C is the number of joint types.
        dets = joint_dets(hmps_hr, self.K)  # [4 * (N, 17, K)], or may be [4 * (N, C, K)]

        n, c, h, w = hmps_hr.shape
        n_limbs = len(self.skeleton)  # L
        LOG.debug('%d limb connections are defined', n_limbs)

        # 2 * (N, L, K, 1)，and (N, L, K, 2); top K limbs of high responses
        kps_inds_f, kps_scores_f, kps_xys_f = self._channel_dets(
            dets, self.jtypes_f, self.thre_hmp)

        # 2 * (N, L, K, 1), can also be 2 * (N, L, M, 1),
        # and (N, L, K, 2), can also be (N, L, M, 2)
        kps_inds_t, kps_scores_t, kps_xys_t = self._channel_dets(
            dets, self.jtypes_t, self.thre_hmp)

        # ########################################################################
        # ############### get keypoint scales for all limb endpoints ############
        # ########################################################################
        if self.include_scale and isinstance(scmps_hr, torch.Tensor):  # if scmps != []
            kps_scales_f = self._channel_scales(
                scmps_hr, kps_inds_f, n, n_limbs, self.jtypes_f)  # # (N, L, K, 1)
            kps_scales_t = self._channel_scales(
                scmps_hr, kps_inds_t, n, n_limbs, self.jtypes_t)  # (N, L, K, 1)
        else:
            kps_scales_f = 4 * torch.ones_like(kps_scores_f,  # (N, L, K, 1)
                                               dtype=kps_scores_f.dtype,
                                               device=kps_scores_f.device)
            kps_scales_t = 4 * torch.ones_like(kps_scores_t,  # (N, L, K, 1)
                                               dtype=kps_scores_t.dtype,
                                               device=kps_scores_t.device)

        # ########################################################################
        # ############### get jitter offset for all limb endpoints ############
        # ########################################################################
        if self.include_jitter_offset and isinstance(jomps_hr, torch.Tensor):  # if scmps != []
            kps_jitter_f = self._channel_jitters(
                jomps_hr, kps_inds_f, n, n_limbs, self.jtypes_f)  # # (N, L, K, 2)
            kps_jitter_t = self._channel_jitters(
                jomps_hr, kps_inds_t, n, n_limbs, self.jtypes_t)  # (N, L, K, 2)
        else:
            kps_jitter_f = torch.zeros(n, n_limbs, self.K, 2,  # (N, L, K, 2)
                                       dtype=kps_xys_f.dtype,
                                       device=kps_xys_f.device)
            kps_jitter_t = torch.zeros(n, n_limbs, self.K, 2,  # (N, L, K, 2)
                                       dtype=kps_xys_t.dtype,
                                       device=kps_xys_t.device)

        # ########################################################################
        # ############### get offset vectors of all limb connections ############
        # ########################################################################
        offs_reshape = offs_hr.view((n, -1, vector_nd, h, w))  # (N, L, 2, H, W)
        flat_off = offs_reshape.view((n, n_limbs, vector_nd, -1))  # stretch and flat to (N, L, 2, H*W)
        kps_inds_f_expand = kps_inds_f.permute((0, 1, 3, 2)).expand(-1, -1, vector_nd, -1)  # (N, L, 2, K)
        # (N, L, 2, H*W) -> (N, L, 2, K) -> (N, L, K, 2)
        kps_off_f = flat_off.gather(-1, kps_inds_f_expand).permute((0, 1, 3, 2))

        # ########################################################################
        # ########## get the regressed end-joints from the start-joints ##########
        # ########################################################################
        kps_guid_t = kps_xys_f.repeat(1, 1, 1, vector_nd // 2) + kps_off_f * self.resize_factor  # (N, L, K, 2)

        # ########################################################################
        # ########### use keypoint jitter offset to refine guiding offset #########
        # ################### but this trick does not help ########################
        # #########################################################################
        if self.use_jitter_offset:
            for i in range(n):
                for j in range(n_limbs):
                    for k in range(self.K):
                        #  be sure to obtain the corresponding refine offset at the kps_guid_t positions
                        xy = kps_guid_t.int()[i, j, k]
                        if 0 <= xy[0] < w and 0 <= xy[1] < h:
                            kps_guid_t[i, j, k] += jomps_hr[i, :, xy[0], xy[1]]

        # ########################################################################
        # ############### find limbs from kps_f_lk to kps_t_lm ###############
        # ########################################################################
        # (N, L, K, M, 2)
        kps_guid_t_expand = kps_guid_t.unsqueeze(3).expand(n, n_limbs, self.K, self.K, vector_nd)
        # (N, L, K, M, 2)
        kps_xys_t_expand = kps_xys_t.repeat(1, 1, 1, vector_nd // 2).unsqueeze(2).expand(n, n_limbs, self.K, self.K,
                                                                                         vector_nd)
        dist = (kps_guid_t_expand - kps_xys_t_expand).norm(dim=-1)  # (N, L, K, M)
        """dist_min ensures one keypoint can only be used and paired once."""
        min_dist, min_ind = dist.min(dim=-1)  # 2 * (N, L, K）
        min_dist = min_dist.unsqueeze(3)  # (N, L, K, 1)
        min_ind = min_ind.unsqueeze(3)  # (N, L, K, 1)

        # ########################################################################
        # ############### get the paired kps_t with regard to kps_f ##############
        # here before, all candidate start-points and end-points are not matched!
        # ########################################################################
        matched_kps_score_t = kps_scores_t.gather(2, min_ind)  # (N, L, K, 1)
        matched_kps_xys_t = kps_xys_t.gather(2, min_ind.expand(
            n, n_limbs, self.K, 2))  # (N, L, K, 2)
        matched_kps_inds_t = kps_inds_t.gather(2, min_ind)  # (N, L, K, 1)
        matched_kps_scales_t = kps_scales_t.gather(2, min_ind)  # (N, L, K, 1)

        # ########################################################################
        # #########  convert to global indexes across heatmap channels ###########
        # ########################################################################
        channel_page_f = torch.tensor(self.jtypes_f, device=kps_inds_f.device). \
            reshape((1, 1, 1, n_limbs)).permute((0, 3, 1, 2))  # (1, L, 1, 1)
        channel_page_t = torch.tensor(self.jtypes_t, device=matched_kps_inds_t.device). \
            reshape((1, 1, 1, n_limbs)).permute((0, 3, 1, 2))
        kps_inds_f = kps_inds_f + channel_page_f * (h * w)  # (N, L, K, 1)
        matched_kps_inds_t = matched_kps_inds_t + channel_page_t * (h * w)  # (N, L, K, 1)

        # ########################################################################
        # ################# length and scores of each candidate limbs ############
        # ########################################################################
        len_limbs = torch.clamp((kps_xys_f - matched_kps_xys_t
                                 ).norm(dim=-1, keepdim=True), min=self.min_len)  # (N, L, K, 1)
        # Is torch.exp(-min_dist / kps_scales_t) more sensible? --No, this leads 0.5 AP drop
        limb_scores = kps_scores_f * matched_kps_score_t * torch.exp(-min_dist / len_limbs)
        # len_limb may be 0, t = min_dist / (len_limbs + 1e-4)

        # ########################################################################
        # ######################## jitter refinement #############################
        # ########################################################################
        matched_kps_jitter_t = kps_jitter_t.gather(2, min_ind.expand(
            n, n_limbs, self.K, 2))  # (N, L, K, 2)
        if self.use_jitter_offset:
            kps_xys_f += kps_jitter_f
            matched_kps_xys_t += matched_kps_jitter_t

        # limbs' shape=(N, L, K, 13), in which the last dim includes:
        # [x1, y1, v1, x2, y2, v2, ind1, ind2, len_delta (min_dist), len_limb, limb_score, scale1, scale2]
        # 0,    1, 2,  3,  4,  5,    6,   7,          8,                9,         10,        11,    12
        limbs = torch.cat((kps_xys_f,
                           kps_scores_f,
                           matched_kps_xys_t,
                           matched_kps_score_t,
                           kps_inds_f.float(),  # for dtype compatibility
                           matched_kps_inds_t.float(),
                           min_dist,
                           len_limbs,
                           limb_scores,
                           kps_scales_f,
                           matched_kps_scales_t), dim=-1)
        LOG.debug('candidate limbs collection time: %.6fs', time.time() - start_time)
        # todo: collect all keypoint indexes to save the isolated keypoints later?
        return limbs

    @staticmethod
    def pack_jtypes(skeleton):
        jtypes_f, jtypes_t = [], []
        for i, (j_f, j_t) in enumerate(skeleton):
            jtypes_f.append(j_f)
            jtypes_t.append(j_t)
        return jtypes_f, jtypes_t

    @staticmethod
    def _channel_dets(dets: tuple, jtypes: list, thresh=0.06) -> tuple:
        # shape of each item of dets: (N, 17, K) or may be (N, C, K), in which K is topK
        dets_channels = [temp[:, jtypes, :].unsqueeze(-1) for temp in dets]
        kps_scores, kps_inds, kps_ys, kps_xs = dets_channels  # [4 * (N, L, K, 1)]
        kps_xys = torch.cat((kps_xs, kps_ys), dim=-1)
        # ######## set candidate keypoints with low responses off the image #######
        kps_xys[kps_scores.expand_as(kps_xys) < thresh] -= 100000
        return kps_inds, kps_scores.float(), kps_xys.float()

    @staticmethod
    def _channel_scales(scsmp, kps_inds, n, n_limbs, jtypes):
        # shape of scsmp: (N, C, H, W)
        scsmp_channels = scsmp[:, jtypes, :, :]  # (N, L, H, W)
        flat_scsmp = scsmp_channels.view((n, n_limbs, -1)).unsqueeze(-1)  # (N, L, H*W, 1)
        kps_scale = flat_scsmp.gather(-2, kps_inds)  # (N, L, K, 1), gathered by kps_inds
        return kps_scale

    @staticmethod
    def _channel_jitters(jomps, kps_inds, n, n_limbs, jtypes):
        # shape of jomps: (N, 2, H, W)
        # shape of kps_inds: (N, L, K, 1)
        assert n_limbs == len(jtypes), 'defined limbs number mismatches'
        jomps_channels = jomps.unsqueeze(1).expand(-1, n_limbs, -1, -1, -1)  # (N, 1, 2, H, W) --> (N, L, 2, H, W)
        flat_jomps = jomps_channels.view((n, n_limbs, 2, -1)).permute(
            (0, 1, 3, 2))  # (N, L, 2,  H*W) --> (N, L, H*W, 2)
        kps_jitter = flat_jomps.gather(-2, kps_inds.expand(-1, -1, -1, 2))  # (N, L, K, 2), gathered by kps_inds
        return kps_jitter
