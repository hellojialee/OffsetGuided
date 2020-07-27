"""Greedily group the keypoints based on guiding (associative) offsets"""
import logging
import time
import random
import torch
import numpy as np
from decoder import hmp_NMS, topK_channel
from config.coco_data import (COCO_KEYPOINTS,
                              COCO_PERSON_SKELETON,
                              COCO_PERSON_WITH_REDUNDANT_SKELETON,
                              REDUNDANT_CONNECTIONS,
                              KINEMATIC_TREE_SKELETON)

LOG = logging.getLogger(__name__)


# TODO：构建factory初始化, 根据args选择skeleton = COCO_PERSON_SKELETON
# TODO: 如果后面有频繁的访问numpy内存操作，改成cython代码会不会更快一些？

class LimbsCollect(object):
    """
    Collect all candidate keypoints and pair them into limbs
    on the basis of guiding offset vectors.

    Attributes:
        hmps_hr (Tensor): with shape (N, C, H, W)
        offs_hr (Tensor): with shape (N, L*2, H, W)
        hmp_s (int): stride of coordinate Unit of heatmaps with respect to that of input image
        off_s (int): stride of coordinate Unit of offetmaps with respect to that of input image
        scsmp (Tensor): (C, H, W), feature map of inferred keypoint scales.
        keypoints (list): keypoints in human skeleton
        skeleton (list): limb sequence, i.e., keypoint connections in the human skeleton
        thre_hmp (float): limb connections below this value are moved outside the image boarder
        topk (int): select the top K responses on each heatmaps
    """

    def __init__(self, hmps_hr, offs_hr, hmp_s, off_s, *, scsmp=[], topk=40, thre_hmp=0.08,
                 keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON):
        self.hmps_hr = hmps_hr  # type: torch.Tensor
        self.offs_hr = offs_hr  # type: torch.Tensor
        self.scsmp = scsmp  # todo: 将利用scale与offset delta大小决定imbs的取舍
        assert hmps_hr.shape[-2:] == offs_hr.shape[-2:], 'spatial resolution should be equal'
        LOG.info('input size of heatmaps: %d * %d, '
                 'input size of offmaps: %d * %d\n '
                 'number of skeleton limbs: %d, '  # separate the long string without comma
                 'response threshold to drop keypoints: threshold=%.4f',
                 hmps_hr.shape[2], hmps_hr.shape[3],
                 offs_hr.shape[2], offs_hr.shape[3],
                 len(skeleton), thre_hmp)
        LOG.debug('hmp stride: %d, off stride: %d ', hmp_s, off_s)
        self.hmp_s = hmp_s
        self.off_s = off_s
        self.resize_factor = off_s / hmp_s
        LOG.info('unify the heatmap coordinate unit and offset coordinate'
                 ' unit using rescale factor %.3f', self.resize_factor)
        self.keypoints = keypoints
        self.skeleton = skeleton
        self.K = topk
        self.thre_hmp = thre_hmp
        self.jtypes_f, self.jtypes_t = self.pack_jtypes(skeleton)

    @staticmethod
    def pack_jtypes(skeleton):
        jtypes_f, jtypes_t = [], []
        for i, (j_f, j_t) in enumerate(skeleton):
            jtypes_f.append(j_f)
            jtypes_t.append(j_t)
        return jtypes_f, jtypes_t

    @staticmethod
    def _channel_dets(dets: tuple, jtypes: list, threshold=0.06) -> tuple:
        dets_channels = [temp[:, jtypes, :].unsqueeze(-1) for temp in dets]
        kps_scores, kps_inds, kps_ys, kps_xs = dets_channels
        kps_xys = torch.cat((kps_xs, kps_ys), dim=-1)
        # ######## set candidate keypoints with low responses off the image #######
        kps_xys[kps_scores.expand_as(kps_xys) < threshold] -= 100000
        return kps_inds, kps_scores, kps_xys

    @staticmethod
    def joint_dets(hmps, k):  # todo: 提出来变成单独的函数
        """Select Top k candidate keypoints in heatmaps"""
        filtered_hmps = hmp_NMS(hmps)
        # shape of hm_score, hm_inds, topk_ys, topk_xs = [batch, 17, topk]
        dets = topK_channel(filtered_hmps, K=k)
        return dets

    def generate_limbs(self):  # todo: convert to fp 16 computation
        """Generate all limbs between adjacent keypoints in the human skeleton tree"""
        # shape of each item of dets: (N, 17, K), in which K equals self.topk
        dets = self.joint_dets(self.hmps_hr, self.K)

        n, c, h, w = self.hmps_hr.shape
        n_limbs = len(self.skeleton)  # L
        LOG.debug('%d limb connections are defined', n_limbs)

        # 2 * (N, L, K, 1)，and (N, L, K, 2)
        kps_inds_f, kps_scores_f, kps_xys_f = self._channel_dets(
            dets, self.jtypes_f, self.thre_hmp)

        # 2 * (N, L, K, 1), can also be 2 * (N, L, M, 1),
        # and (N, L, K, 2), can also be (N, L, M, 2)
        kps_inds_t, kps_scores_t, kps_xys_t = self._channel_dets(
            dets, self.jtypes_t, self.thre_hmp)

        # ########################################################################
        # ############### get offset vectors of all limb connections ############
        # ########################################################################
        offs_i = self.offs_hr.view((n, -1, 2, h, w))  # (N, L, 2, H, W)
        flat_off_i = offs_i.view((n, n_limbs, 2, -1))  # stretch and flat to (N, L, 2, H*W)
        kps_inds_f_expand = kps_inds_f.permute((0, 1, 3, 2)).expand(-1, -1, 2, -1)  # (N, L, 2, K)
        # (N, L, 2, K) -> (N, L, K, 2)
        kps_off_f = flat_off_i.gather(-1, kps_inds_f_expand).permute((0, 1, 3, 2))

        # ########################################################################
        # ############### get the regressed end-joints from the start-joints #########
        # ########################################################################
        kps_guid_t = kps_xys_f + kps_off_f * self.resize_factor  # (N, L, K, 2)

        # ########################################################################
        # ############### find limbs from kps_f_lk to kps_t_lm ###############
        # ########################################################################
        # (N, L, K, M, 2)
        kps_guid_t_expand = kps_guid_t.unsqueeze(3).expand(n, n_limbs, self.K, self.K, 2)
        # (N, L, K, M, 2)
        kps_xys_t_expand = kps_xys_t.unsqueeze(2).expand(n, n_limbs, self.K, self.K, 2)
        dist = (kps_guid_t_expand - kps_xys_t_expand).norm(dim=-1)  # (N, L, K, M)
        """dist_min ensures one keypoint can only be used and paired once."""
        min_dist, min_ind = dist.min(dim=-1)  # 2 * (N, L, K）
        min_dist = min_dist.unsqueeze(3)  # (N, L, K, 1)
        min_ind = min_ind.unsqueeze(3)  # (N, L, K, 1)

        # ########################################################################
        # ############### get the paired kps_t with regard to kps_f ##############
        # ########################################################################
        matched_kps_score_t = kps_scores_t.gather(2, min_ind)  # (N, L, K, 1)
        matched_kps_xys_t = kps_xys_t.gather(2, min_ind.expand(
            n, n_limbs, self.K, 2))  # (N, L, K, 2)
        matched_kps_inds_t = kps_inds_t.gather(2, min_ind)  # (N, L, K, 1)

        # ########################################################################
        # #########  convert to global indexes across heatmap channels ###########
        # ########################################################################
        channel_page_f = torch.tensor(self.jtypes_f, device=kps_inds_f.device). \
            reshape((1, 1, 1, n_limbs)).permute((0, 3, 1, 2))  # (1, L, 1, 1)
        channel_page_t = torch.tensor(self.jtypes_t, device=matched_kps_inds_t.device). \
            reshape((1, 1, 1, n_limbs)).permute((0, 3, 1, 2))

        kps_inds_f = kps_inds_f + channel_page_f * (h * w)  # (N, L, K, 1)
        matched_kps_inds_t = matched_kps_inds_t + channel_page_t * (h * w)  # (N, L, K, 1)

        len_limbs = (kps_xys_f.float() - matched_kps_xys_t.float()
                     ).norm(dim=-1, keepdim=True)  # (N, L, K, 1)

        limbs = torch.cat((kps_xys_f.float(),
                           kps_scores_f,
                           matched_kps_xys_t.float(),
                           matched_kps_score_t,
                           kps_inds_f.float(),
                           matched_kps_inds_t.float(),
                           min_dist,
                           len_limbs), dim=-1)

        # shape=(N, L, K, 10), in which the last dim includes:
        # [x1, y1, v1, x2, y2, v2, ind1, ind2, len_delta, len_limb]
        # len_limb may be 0
        #  t = min_dist / (len_limbs + 1e-4)

        return limbs.cpu()  # limbs.cpu().numpy()  # limbs TODO：测试一下哪种更快？？


class GreedyGroup(object):
    """
    Greedily group the limbs into individual human skeletons in one image.
    Args:
        limbs (Tensor): (L, K, 10), includes all limbs in the same image.
        threshold (float): threshold for pose instance scores
    """

    def __init__(self, limbs, threshold, *, dist_max=10,
                 keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON):
        self.limbs = limbs
        self.skeleton = skeleton
        self.keypoints = keypoints
        self.dist_max = dist_max
        self.n_keypoints = len(keypoints)
        self.threshold = threshold

    def group_skeletons(self, force_complete=False):
        assert len(self.limbs) == len(self.skeleton
                                      ), 'check the skeleton config and input limbs Tensor'
        # subset shape is (1, 17, 4), mat be (M, 17, 4), the last dim includes [x, y, v, dist, ind]
        subset = -1 * torch.ones((1, self.n_keypoints, 5))  # 注意 默认的dist竟然为-1...
        subset[:, :, -2] = 10000  # set default dist to a big number

        # Loop over all kinds of limb types
        for i, ((jtype_f, jtype_t), conns) in enumerate(zip(self.skeleton, self.limbs)):
            LOG.debug('limbs from jtype_f %d --> jtype_t %d', jtype_f, jtype_t)

            dist_valid = conns[:, 8] < self.dist_max  # todo: change the dist to element-wise keypoint scales
            valid = dist_valid & (conns[:, 0] > 0) & (conns[:, 3] > 0)
            keep_inds, = valid.nonzero(as_tuple=True)
            if keep_inds.numel() == 0:
                continue
            conns = conns[keep_inds]  # (K, 10), may be (kk, 10) in which kk<K

            jIDtab = subset[:, [jtype_f, jtype_t], -1]  # type: torch.Tensor # (M, 2)
            distab = subset[:, [jtype_f, jtype_t], -2]  # type: torch.Tensor # (M, 2)

            xyv1 = conns[:, :3]  # (K, 3)
            xyv2 = conns[:, 3:6]  # (K, 3)
            limb_inds = conns[:, 6:8]  # (K, 2), joint_f_ID and joint_t_ID

            limb_lens = conns[:, 8:]  # (K, 2), delta_length, limb_length

            # suppose there are M pose skeletons, then the expanded shape is (M, K, 2)
            jIDtab_expand = jIDtab.unsqueeze(1).expand(-1, keep_inds.numel(), -1)  # (M, K, 2)
            distab_expand = distab.unsqueeze(1).expand(-1, keep_inds.numel(), -1)  # (M, K, 2)

            limb_inds_expand = limb_inds.unsqueeze(0).expand_as(jIDtab_expand)  # (M, K, 2)
            limb_lens_expand = limb_lens.unsqueeze(0).expand_as(distab_expand)  # (M, K, 2)

            mask_sum = (jIDtab_expand.int() == limb_inds_expand.int()).int().sum(dim=-1)  # (M, K)
            # criterion to judge if we replace the exiting keypoints
            dist_mask = (limb_lens_expand[:, :, 0] / (limb_lens_expand[:, :, 0] + 0.1)
                         < distab_expand[:, :, 1])  # (M, K), scale of joint_t

            # ########################################################################
            # ########################## generate new skeletons ######################
            # ########################################################################
            # set as_tuple=True will return a tuple of 1-D indexes, i.e., (tensor, )
            New_inds, = (mask_sum.sum(dim=0) == 0).nonzero(as_tuple=True)
            if New_inds.numel():
                rows = -1 * torch.ones((len(New_inds), self.n_keypoints, 5))
                rows[:, [jtype_f, jtype_t], -1] = limb_inds[New_inds]
                rows[:, jtype_f, :3] = xyv1[New_inds]
                rows[:, jtype_t, :3] = xyv2[New_inds]
                # initial two connected keypoint share the same limb delta
                rows[:, jtype_f, 3] = limb_lens[New_inds, 0]
                rows[:, jtype_t, 3] = limb_lens[New_inds, 0]
                subset = torch.cat((subset, rows), dim=0)

            # ########################################################################
            # ############## connect current limbs with existing skeletons ###########
            # ########################################################################
            #  & dist_mask # fixme: if we use dist_mask, small keypoint may be not added
            # change to normalized dist by limb-length, this issue still not solved
            M_inds, K_inds = (mask_sum == 1).nonzero(as_tuple=True)
            if M_inds.numel() and K_inds.numel():
                subset[M_inds, jtype_f, -1] = limb_inds[K_inds, 0]
                subset[M_inds, jtype_t, -1] = limb_inds[K_inds, 1]

                subset[M_inds, jtype_f, :3] = xyv1[K_inds]
                subset[M_inds, jtype_t, :3] = xyv2[K_inds]
                # subset[M_inds, jtype_f, 3] = limb_lens[K_inds, 0]  # should not change jtype_f scale
                subset[M_inds, jtype_t, -2] = limb_lens[K_inds, 0]

            # ########################################################################
            # ######### merge the subset belonging to the same person skeleton #######
            # ########################################################################
            # M_inds, K_inds = ((mask_sum == 2) & ).nonzero(as_tuple=True)
        t = subset.numpy()
        print(t.shape)

