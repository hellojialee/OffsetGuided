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
# TODO: 如果后面有频繁的访问numpy内存操作，改成cython代码会不会更快一些？

class LimbsCollect(object):
    """
    Collect all **candidate** keypoints and pair them into limbs
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
        min_len (int): length in pixels, in case of the zero length of limb
    """

    def __init__(self, hmps_hr, offs_hr, hmp_s, off_s, *, scsmp=[], topk=40, thre_hmp=0.08,
                 min_len=3, keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON):
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
        self.min_len = min_len
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

    def generate_limbs(self):
        """
        Generate all limbs between adjacent keypoints in the human skeleton tree

        Returns: a Tensor containing all candidate limbs information of a batch of images
        """
        # shape of each item of dets: (N, 17, K), in which K equals self.topk
        dets = joint_dets(self.hmps_hr, self.K)

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

        len_limbs = torch.clamp((kps_xys_f.float() - matched_kps_xys_t.float()
                                 ).norm(dim=-1, keepdim=True), min=self.min_len)  # (N, L, K, 1)

        limb_scores = kps_scores_f * matched_kps_score_t * torch.exp(-min_dist / len_limbs)
        limbs = torch.cat((kps_xys_f.float(),
                           kps_scores_f,
                           matched_kps_xys_t.float(),
                           matched_kps_score_t,
                           kps_inds_f.float(),
                           matched_kps_inds_t.float(),
                           min_dist,
                           len_limbs,
                           limb_scores), dim=-1)

        # shape=(N, L, K, 11), in which the last dim includes:
        # [x1, y1, v1, x2, y2, v2, ind1, ind2, len_delta, len_limb, limb_score]
        # len_limb may be 0
        #  t = min_dist / (len_limbs + 1e-4)

        return limbs.cpu().numpy()


class GreedyGroup(object):
    """
    Greedily group the limbs into individual human skeletons in one image.
    Args:
        limbs (np.ndarray): (L, K, 10), includes all limbs in the same image.
        threshold (float): threshold for pose instance scores.
        dist_max (float): abandon limbs with delta offsets bigger than dist_max。
    """

    def __init__(self, limbs, threshold, *, dist_max=10,
                 keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON):
        self.limbs = limbs
        self.skeleton = skeleton
        self.keypoints = keypoints
        self.dist_max = dist_max
        self.n_keypoints = len(keypoints)
        self.threshold = threshold
        assert len(self.limbs) == len(self.skeleton
                                      ), 'check the skeleton config and input limbs Tensor'

    def group_skeletons(self, force_complete=False):
        # subset shape is (0, 17, 4), mat be (M, 17, 4), the last dim includes [x, y, v, limb_score, ind]
        subset = -1 * np.ones((0, self.n_keypoints, 5))

        # Loop over all kinds of Limb types
        for i, ((jtype_f, jtype_t), conns) in enumerate(zip(self.skeleton, self.limbs)):
            LOG.debug('limbs from jtype_f %d --> jtype_t %d', jtype_f, jtype_t)

            # todo: change the dist to element-wise keypoint scales
            dist_valid = conns[:, 8] < self.dist_max

            valid = dist_valid & (conns[:, 0] > 0) & (conns[:, 3] > 0) & (
                    conns[:, 1] > 0) & (conns[:, 4] > 0)
            conns = conns[valid]  # (K, 11), may be (kk, 11) in which kk<K

            # ############ delete limb connections sharing the same keypoint ############
            conns = self._delete_reconns(conns)

            if len(conns) == 0:
                continue

            jIDtab = subset[:, [jtype_f, jtype_t], -1]  # (M, 2)
            sub_scores = subset[:, [jtype_f, jtype_t], -2]  # (M, 2)

            xyv1 = conns[:, :3]  # (K, 3)
            xyv2 = conns[:, 3:6]  # (K, 3)
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
                subset[M_inds, jtype_f, 3] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_f, 3])
                subset[M_inds, jtype_t, 3] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_t, 3])
                mask_sum[mask_sum == 2] = -1  # mask out the solved limbs

            # ########################################################################
            # ############## connect current limbs with existing skeletons ###########
            # ########################################################################
            M_inds, K_inds = ((mask_sum == 1) & replace_mask).nonzero()
            if len(M_inds):
                subset[M_inds, jtype_f, -1] = limb_inds[K_inds, 0]
                subset[M_inds, jtype_t, -1] = limb_inds[K_inds, 1]
                subset[M_inds, jtype_f, :3] = xyv1[K_inds]
                subset[M_inds, jtype_t, :3] = xyv2[K_inds]
                # maybe the current limb shares the joint_f OR joint_t with some person skeleton
                subset[M_inds, jtype_f, 3] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_f, 3])
                subset[M_inds, jtype_t, 3] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_t, 3])
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
                M_inds, N_inds = (merge_mask_sum == 2).nonzero()  # todo: 添加merge的阈值条件？
                if len(M_inds):  # merge skeletons belonging to the same person
                    subset[M_inds, :, :] = np.maximum(subset[M_inds, :, :], subset[N_inds, :, :])
                    subset = np.delete(subset, N_inds, axis=0)

                # other cases
                M_inds, N_inds = (merge_mask_sum >= 3).nonzero()
                if len(M_inds):
                    print('usually this never happens, we ignore handling skeletons crossing at 3 joints')
                    pass

            # ########################################################################
            # ########################## generate new skeletons ######################
            # ########################################################################
            New_inds, = (np.sum(mask_sum, axis=0) == 0).nonzero()  # sum(tensor of size[0])=0
            if len(New_inds):
                rows = -1 * np.ones((len(New_inds), self.n_keypoints, 5))
                rows[:, [jtype_f, jtype_t], -1] = limb_inds[New_inds]
                rows[:, jtype_f, :3] = xyv1[New_inds]
                rows[:, jtype_t, :3] = xyv2[New_inds]
                # initial two connected keypoint share the same limb delta
                rows[:, jtype_f, 3] = limb_scores[New_inds, 1]
                rows[:, jtype_t, 3] = limb_scores[New_inds, 1]
                subset = np.concatenate((subset, rows), axis=0)

            if force_complete:
                # todo: 将没有limb连接分配的且响应高的点强行分配，但是貌似做不到，
                #  因为这依赖于之前生成的candidate limbs，总是两个joints对
                pass
        # t = subset  # for debug
        # print(subset.shape)

        return subset  # numpy array [M * [x, y, v, limb_score, ind]]

    @staticmethod
    def _delete_reconns(conns):
        """
        Ensure one keypoint can only be used once by adjacent keypoints in the skeleton.
        Args:
            conns (np.ndarray): shape (K, 11), in which the last dim includes:
                [x1, y1, v1, x2, y2, v2, ind1, ind2, len_delta, len_limb, limb_score]
        """
        conns = conns[np.argsort(-conns[:, -1])]  # sort by subset_scores
        repeat_check = []
        unique_list = []
        for j, ind_t in enumerate(conns[:, 7].astype(int)):
            if ind_t not in repeat_check:
                repeat_check.append(ind_t)
                unique_list.append(j)
        conns = conns[unique_list]
        return conns
