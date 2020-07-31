"""Greedily group the keypoints based on guiding (associative) offsets"""
import logging
import time
import random
import torch
import numpy as np
from decoder import joint_dets
from config.coco_data import (COCO_KEYPOINTS,
                              COCO_PERSON_SKELETON,
                              COCO_PERSON_WITH_REDUNDANT_SKELETON,
                              REDUNDANT_CONNECTIONS,
                              KINEMATIC_TREE_SKELETON)

LOG = logging.getLogger(__name__)


# TODO：构建factory初始化, 根据args选择skeleton = COCO_PERSON_SKELETON

class LimbsCollect(object):
    """
    Collect all **candidate** keypoints and pair them into limbs
    on the basis of guiding offset vectors.

    Drawback: can not deal with isolated keypoints well.

    Attributes:
        hmp_s (int): stride of coordinate Unit of heatmaps with respect to that of input image
        off_s (int): stride of coordinate Unit of offetmaps with respect to that of input image
        keypoints (list): keypoints in human skeleton
        topk (int): select the top K responses on each heatmaps
        min_len (int): length in pixels, in case of the zero length of limb
        use_scale (bool): use the inferred keypoint scales.
        skeleton (list): limb sequence, i.e., keypoint connections in the human skeleton
        thre_hmp (float): candidate kepoints below this value are moved outside the image boarder
    """

    def __init__(self, hmp_s, off_s, *, topk=40, thre_hmp=0.08,
                 min_len=3, use_scale=False, keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON):
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
        self.use_scale = use_scale
        self.jtypes_f, self.jtypes_t = self.pack_jtypes(skeleton)

    def generate_limbs(self,
                       hmps_hr: torch.Tensor,
                       offs_hr: torch.Tensor,
                       scsmp=None) -> np.ndarray:
        """
        Generate all candidate limbs between adjacent keypoints in the human skeleton tree,
         on the basis of the regressed heatmaps and offsetmaps

        Args:
            hmps_hr (Tensor): with shape (N, C, H, W)
            offs_hr (Tensor): with shape (N, L*2, H, W)
            scsmp (Tensor): (N, C, H, W), feature map of inferred keypoint scales.

        Returns: a Tensor containing all candidate limbs information of a batch of images
        """
        assert hmps_hr.shape[-2:] == offs_hr.shape[-2:], 'spatial resolution should be equal'

        LOG.debug('input size of heatmaps: %d * %d, '
                  'input size of offmaps: %d * %d ',
                  hmps_hr.shape[2], hmps_hr.shape[3],
                  offs_hr.shape[2], offs_hr.shape[3])
        start_time = time.time()
        # ########################################################################
        # ################ get all information for all limb endpoints ############
        # ########################################################################
        # shape of each item of dets: (N, 17, K) or may be (N, C, K),
        # in which K equals self.K and C is the number of joint types.
        dets = joint_dets(hmps_hr, self.K)  # [4 * (N, 17, K)], or may be [4 * (N, C, K)]

        n, c, h, w = hmps_hr.shape
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
        # ############### get keypoint scales for all limb endpoints ############
        # ########################################################################
        if self.use_scale and (scsmp is not None):
            kps_scales_f = self._channel_scales(
                scsmp, kps_inds_f, n, n_limbs, self.jtypes_f)  # # (N, L, K, 1)
            kps_scales_t = self._channel_scales(
                scsmp, kps_inds_t, n, n_limbs, self.jtypes_t)  # (N, L, K, 1)
        else:
            kps_scales_f = 4 * torch.ones_like(kps_inds_f,   # (N, L, K, 1)
                                               dtype=torch.float,
                                               device=kps_inds_f.device)
            kps_scales_t = 4 * torch.ones_like(kps_inds_t,  # (N, L, K, 1)
                                               dtype=torch.float,
                                               device=kps_inds_t.device)

        # ########################################################################
        # ############### get offset vectors of all limb connections ############
        # ########################################################################
        offs_reshape = offs_hr.view((n, -1, 2, h, w))  # (N, L, 2, H, W)
        flat_off = offs_reshape.view((n, n_limbs, 2, -1))  # stretch and flat to (N, L, 2, H*W)
        kps_inds_f_expand = kps_inds_f.permute((0, 1, 3, 2)).expand(-1, -1, 2, -1)  # (N, L, 2, K)
        # (N, L, 2, K) -> (N, L, K, 2)
        kps_off_f = flat_off.gather(-1, kps_inds_f_expand).permute((0, 1, 3, 2))

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
        len_limbs = torch.clamp((kps_xys_f.float() - matched_kps_xys_t.float()
                                 ).norm(dim=-1, keepdim=True), min=self.min_len)  # (N, L, K, 1)
        # todo: 使用torch.exp(-min_dist / kps_scales_t)会不会更合理？
        limb_scores = kps_scores_f * matched_kps_score_t * torch.exp(-min_dist / len_limbs)
        # len_limb may be 0, t = min_dist / (len_limbs + 1e-4)
        # limbs' shape=(N, L, K, 13), in which the last dim includes:
        # [x1, y1, v1, x2, y2, v2, ind1, ind2, len_delta (min_dist), len_limb, limb_score, scale1, scale2]
        # 0,    1, 2,  3,  4,  5,    6,   7,          8,                9,         10,        11,    12
        limbs = torch.cat((kps_xys_f.float(),
                           kps_scores_f,
                           matched_kps_xys_t.float(),
                           matched_kps_score_t,
                           kps_inds_f.float(),
                           matched_kps_inds_t.float(),
                           min_dist,
                           len_limbs,
                           limb_scores,
                           kps_scales_f,
                           matched_kps_scales_t), dim=-1)
        LOG.debug('candidate limbs collection time: %.6fs', time.time() - start_time)
        # todo: collect all keypoint indexes to save the isolated keypoints later
        return limbs.cpu().numpy()

    @staticmethod
    def pack_jtypes(skeleton):
        jtypes_f, jtypes_t = [], []
        for i, (j_f, j_t) in enumerate(skeleton):
            jtypes_f.append(j_f)
            jtypes_t.append(j_t)
        return jtypes_f, jtypes_t

    @staticmethod
    def _channel_dets(dets: tuple, jtypes: list, threshold=0.06) -> tuple:
        # shape of each item of dets: (N, 17, K) or may be (N, C, K), in which K is topK
        dets_channels = [temp[:, jtypes, :].unsqueeze(-1) for temp in dets]
        kps_scores, kps_inds, kps_ys, kps_xs = dets_channels  # [4 * (N, L, K, 1)]
        kps_xys = torch.cat((kps_xs, kps_ys), dim=-1)
        # ######## set candidate keypoints with low responses off the image #######
        kps_xys[kps_scores.expand_as(kps_xys) < threshold] -= 100000
        return kps_inds, kps_scores, kps_xys

    @staticmethod
    def _channel_scales(scsmp, kps_inds, n, n_limbs, jtypes):
        # shape of scsmp: (N, C, H, W)
        scsmp_channels = scsmp[:, jtypes, :, :]  # (N, L, H, W)
        flat_scsmp = scsmp_channels.view((n, n_limbs, -1)).unsqueeze(-1)  # (N, L, H*W, 1)
        kps_scale = flat_scsmp.gather(-2, kps_inds)  # (N, L, K, 1)
        return kps_scale


class GreedyGroup(object):
    """
    Greedily group the limbs into individual human skeletons in one image.

    Args:
        person_thre (float): threshold for pose instance scores.
        del_sort (bool): delete and sort the detected person poses.
        dist_max (float): abandon limbs with delta offsets bigger than dist_max, if joint scales are unavailable.
        use_scale (bool): use the inferred keypoint scales.
    """

    def __init__(self, person_thre, *, del_sort=True, dist_max=10, use_scale=False,
                 keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON):
        self.use_scale = use_scale
        self.del_sort = del_sort
        self.skeleton = skeleton
        self.keypoints = keypoints
        self.dist_max = dist_max
        self.n_keypoints = len(keypoints)
        self.person_thre = person_thre

    def group_skeletons(self, limbs, force_complete=False):
        """
        Group all candidate limbs into individual human poses.

        Args:
        limbs (np.ndarray): (L, K, 10), includes all limbs in the same image.
        """
        assert len(limbs) == len(self.skeleton
                                 ), 'check the skeleton config and input limbs Tensor'
        # subset shape is (0, 17, 6), mat be (M, 17, 6),
        # the last dim includes [x, y, v, s, limb_score, ind]
        start_time = time.time()
        subset = -1 * np.ones((0, self.n_keypoints, 6))

        # Loop over all kinds of Limb types
        for i, ((jtype_f, jtype_t), conns) in enumerate(zip(self.skeleton, limbs)):
            LOG.debug('limbs from jtype_f %d --> jtype_t %d', jtype_f, jtype_t)

            if self.use_scale:  # conns: (K, 12)
                dist_valid = conns[:, 8] < np.maximum(self.dist_max, conns[:, 12])  # 12: joint_t scale
            else:  # conns: (K, 10)
                dist_valid = conns[:, 8] < self.dist_max
                # conns = np.hstack((conns, 4 * np.ones((len(conns), 2))))  # we did this in LimbsCollect

            # ######################## remove false limbs ##############################
            valid = dist_valid & (conns[:, 0] > 0) & (conns[:, 3] > 0) & (
                    conns[:, 1] > 0) & (conns[:, 4] > 0)
            conns = conns[valid]  # (K, 11), may be (kk, 11) in which kk<K

            # ############ delete limb connections sharing the same keypoint ############
            conns = self._delete_reconns(conns)

            if len(conns) == 0:
                continue

            jIDtab = subset[:, [jtype_f, jtype_t], -1]  # (M, 2)
            sub_scores = subset[:, [jtype_f, jtype_t], -2]  # (M, 2)

            xyvs1 = conns[:, [0, 1, 2, 11]]  # (K, 4)
            xyvs2 = conns[:, [3, 4, 5, 12]]  # (K, 4)
            limb_inds = conns[:, 6:8]  # (K, 2), joint_f_ID and joint_t_ID
            limb_scores = conns[:, [8, 10]]  # (K, 2), delta_length, limb_scores

            # suppose there are M pose skeletons, then the expanded shape is (M, K, 2)
            kk = len(conns)
            mm = len(subset)
            jIDtab_expand = np.expand_dims(jIDtab, axis=1).repeat(kk, axis=1)  # (M, K, 2)
            sub_scores_expand = np.expand_dims(sub_scores, axis=1).repeat(kk, axis=1)  # (M, K, 2)

            limb_inds_expand = np.expand_dims(limb_inds, axis=0).repeat(mm, axis=0)  # (M, K, 2)
            limb_scores_expand = np.expand_dims(limb_scores, axis=0).repeat(mm, axis=0)  # (M, K, 2)

            mask_sum = np.sum((jIDtab_expand.astype(int) == limb_inds_expand.astype(int)),
                              axis=-1)  # (M, K)

            # criterion to judge if we replace the exiting keypoints
            replace_mask = (limb_scores_expand[..., 1] > sub_scores_expand[..., 1]) | (
                    limb_scores_expand[..., 1] > sub_scores_expand[..., 0])  # (M, K), score of joint_t

            # ########################################################################
            # ######### handle redundant limbs belonging to the same person skeleton #######
            # ########################################################################
            M_inds, K_inds = ((mask_sum == 2) & replace_mask).nonzero()  # do not forget the parenthesis!
            if len(M_inds):
                # maybe the current limb shares the joint_f OR joint_t with some person skeleton
                subset[M_inds, jtype_f, 4] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_f, 4])
                subset[M_inds, jtype_t, 4] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_t, 4])
                mask_sum[mask_sum == 2] = -1  # mask out the solved limbs

            # ########################################################################
            # ############## connect current limbs with existing skeletons ###########
            # ########################################################################
            M_inds, K_inds = ((mask_sum == 1) & replace_mask).nonzero()
            if len(M_inds):
                subset[M_inds, jtype_f, -1] = limb_inds[K_inds, 0]
                subset[M_inds, jtype_t, -1] = limb_inds[K_inds, 1]
                subset[M_inds, jtype_f, :4] = xyvs1[K_inds]
                subset[M_inds, jtype_t, :4] = xyvs2[K_inds]
                # maybe the current limb shares the joint_f OR joint_t with some person skeleton
                subset[M_inds, jtype_f, 4] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_f, 4])
                subset[M_inds, jtype_t, 4] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_t, 4])
                mask_sum[mask_sum == 1] = -1  # mask out the solved limbs

            # ########################################################################
            # ######### merge the subsets belonging to the same person skeleton #######
            # ########################################################################
            if mm >= 2:
                Msubset_expand = np.expand_dims(subset[..., -1], axis=1).repeat(mm, axis=1)  # (M, M, 17) or (M, N, 17)
                Nsubset_expand = np.expand_dims(subset[..., -1], axis=0).repeat(mm, axis=0)  # (M, M, 17) or (M, N, 17)
                merge_mask_sum = np.sum((Msubset_expand.astype(int) == Nsubset_expand.astype(int))
                                        & (Msubset_expand.astype(int) != -1),  # & (Nsubset_expand.astype(int) != -1)
                                        axis=-1)  # (M, M)
                # np.fill_diagonal(merge_mask_sum, 0)
                merge_mask_sum = np.triu(merge_mask_sum, 1)
                M_inds, N_inds = (merge_mask_sum == 2).nonzero()
                if len(M_inds):  # merge skeletons belonging to the same person
                    # overlay -1 elements or small scores, and keep the same keypoint info
                    subset[M_inds, :, :] = np.maximum(subset[M_inds, :, :], subset[N_inds, :, :])
                    subset = np.delete(subset, N_inds, axis=0)

                # other cases
                M_inds, N_inds = (merge_mask_sum >= 3).nonzero()
                if len(M_inds):
                    print('usually this never happens, ignore handling skeletons crossing at 3 joints')
                    pass

            # ########################################################################
            # ########################## generate new skeletons ######################
            # ########################################################################
            New_inds, = (np.sum(mask_sum, axis=0) == 0).nonzero()  # sum(tensor of size[0])=0
            if len(New_inds):
                rows = -1 * np.ones((len(New_inds), self.n_keypoints, 6))
                rows[:, [jtype_f, jtype_t], -1] = limb_inds[New_inds]
                rows[:, jtype_f, :4] = xyvs1[New_inds]
                rows[:, jtype_t, :4] = xyvs2[New_inds]
                # initial two connected keypoint share the same limb_score
                rows[:, jtype_f, 4] = limb_scores[New_inds, 1]
                rows[:, jtype_t, 4] = limb_scores[New_inds, 1]
                subset = np.concatenate((subset, rows), axis=0)

        LOG.debug('keypoint grouping time in the current image: %.6fs', time.time() - start_time)

        if self.del_sort:
            subset = self._delete_sort(subset, self.person_thre)

        return subset  # numpy array [M * [ 17 * [x, y, v, s, limb_score, ind]]]

    @staticmethod
    def _delete_sort(subset, thre):
        t0 = time.time()
        delete_list = []
        scores_list = []
        for i in range(len(subset)):
            kps_mask = subset[i, :, 2] > 0
            score = subset[i, kps_mask, 2].sum() / kps_mask.astype(np.float).sum()
            if score < thre:
                delete_list.append(i)
            else:
                scores_list.append(score)
        subset = np.delete(subset, delete_list, axis=0)
        sort_inds = sorted(range(len(scores_list)), key=lambda k: scores_list[k], reverse=True)
        subset = subset[sort_inds]
        LOG.debug('delete and sort the poses: %.6fs', time.time() - t0)
        return subset

    @staticmethod
    def _delete_reconns(conns):
        """
        Ensure one keypoint can only be used once by adjacent keypoints in the skeleton.
        Args:
            conns (np.ndarray): shape (K, 11), in which the last dim includes:
                [x1, y1, v1, x2, y2, v2, ind1, ind2, len_delta, len_limb, limb_score, scale1, scale2]
        """
        conns = conns[np.argsort(-conns[:, -1])]  # -1: sort by limb_scores
        repeat_check = []
        unique_list = []
        for j, ind_t in enumerate(conns[:, 7].astype(int)):
            if ind_t not in repeat_check:
                repeat_check.append(ind_t)
                unique_list.append(j)
        conns = conns[unique_list]
        return conns
