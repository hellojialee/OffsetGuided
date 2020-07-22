"""Greedily group the keypoints based on guid (associative) offsets"""
import logging
import time
import torch
import numpy as np
from decoder import hmp_NMS, topK_channel

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
        skeleton (list): limb sequence, i.e., keypoint connections in the human skeleton
        topk (int): select the top K responses on each heatmaps,
            thus top K limb connections are generated
    """

    def __init__(self, hmps_hr, offs_hr, hmp_s, off_s, skeleton, topk=40):
        self.hmps_hr = hmps_hr  # type: torch.Tensor
        self.offs_hr = offs_hr  # type: torch.Tensor
        assert hmps_hr.shape[-2:] == offs_hr.shape[-2:], 'spatial resolution should be equal'
        LOG.info('input size of heatmaps: %d * %d, '
                 'input size of offmaps: %d * %d',
                 hmps_hr.shape[2], hmps_hr.shape[3],
                 offs_hr.shape[2], offs_hr.shape[3])
        LOG.debug('hmp stride: %d, off stride: %d ', hmp_s, off_s)
        self.hmp_s = hmp_s
        self.off_s = off_s
        self.resize_factor = off_s / hmp_s
        LOG.info('unify the heatmap coordinate unit and offset coordinate'
                 ' unit using rescale factor %.3f', self.resize_factor)
        self.skeleton = skeleton
        self.K = topk
        self.jtypes_f, self.jtypes_t = self.pack_jtypes()

    @staticmethod
    def joint_dets(hmps, k):
        """Select Top k candidate keypoints in heatmaps"""
        filtered_hmps = hmp_NMS(hmps)
        # shape of hm_score, hm_inds = [batch, 17, topk]
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
        kps_inds_f, kps_scores_f, kps_xys_f = self._channel_dets(dets, self.jtypes_f)

        # 2 * (N, L, K, 1), can also be (N, L, M, 1)
        # and (N, L, K, 2), can also be (N, L, M, 2)
        kps_inds_t, kps_scores_t, kps_xys_t = self._channel_dets(dets, self.jtypes_t)

        # ############### get offset vectors of all limb connections ###############
        offs_i = self.offs_hr.view((n, -1, 2, h, w))  # (N, L, 2, H, W)
        flat_off_i = offs_i.view((n, n_limbs, 2, -1))  # stretch and flat to (N, L, 2, H*W)
        kps_inds_f_expand = kps_inds_f.permute((0, 1, 3, 2)).expand(-1, -1, 2, -1)  # (N, L, 2, K)
        # (N, L, 2, K) -> (N, L, K, 2)
        kps_off_f = flat_off_i.gather(-1, kps_inds_f_expand).permute((0, 1, 3, 2))

        # ############### get the regressed end-joints from the start-joints ###############
        kps_guid_t = kps_xys_f + kps_off_f * self.resize_factor  # (N, L, K, 2)
        # ############### find limbs from kps_f_lk to kps_t_lm ###############
        # (N, L, K, M, 2)
        kps_guid_t_expand = kps_guid_t.unsqueeze(3).expand(n, n_limbs, self.K, self.K, 2)
        # (N, L, K, M, 2)
        kps_xys_t_expand = kps_xys_t.unsqueeze(2).expand(n, n_limbs, self.K, self.K, 2)
        dist = (kps_guid_t_expand - kps_xys_t_expand).norm(dim=-1)  # (N, L, K, M)
        min_dist, min_ind = dist.min(dim=-1)  # 2 * (N, L, K）
        min_dist = min_dist.unsqueeze(3)  # (N, L, K, 1)
        min_ind = min_ind.unsqueeze(3)  # (N, L, K, 1)

        # ############### get the paired kps_t with regard to kps_f ###############
        matched_kps_score_t = kps_scores_t.gather(2, min_ind)  # (N, L, K, 1)
        matched_kps_xys_t = kps_xys_t.gather(2, min_ind.expand(n, n_limbs, self.K, 2))  # (N, L, K, 2)
        matched_kps_inds_t = kps_inds_t.gather(2, min_ind)  # (N, L, K, 1)

        # ###############  convert to global indexes across heatmap channels ###############
        channel_page_f = torch.tensor(self.jtypes_f, device=kps_inds_f.device). \
            reshape((1, 1, 1, n_limbs)).permute((0, 3, 1, 2))  # (1, L, 1, 1)
        channel_page_t = torch.tensor(self.jtypes_t, device=matched_kps_inds_t.device). \
            reshape((1, 1, 1, n_limbs)).permute((0, 3, 1, 2))

        kps_inds_f = kps_inds_f + channel_page_f * (h * w)  # (N, L, K, 1)
        matched_kps_inds_t = matched_kps_inds_t + \
                             channel_page_t * (h * w)  # (N, L, K, 1)

        len_limbs = (kps_xys_f.float() - matched_kps_xys_t.float()).norm(dim=-1, keepdim=True)  # (N, L, K, 1)

        limbs = torch.cat((kps_xys_f.float(),
                           kps_scores_f,
                           kps_inds_f.float(),
                           matched_kps_xys_t.float(),
                           matched_kps_score_t,
                           matched_kps_inds_t.float(),
                           min_dist,
                           len_limbs), -1)

        # shape=(N, L, K, 10), in the last dim: [x1, y1, v1, ind1, x2, y2, v2, ind2, len_delta, len_limb]
        return limbs.cpu().numpy()

    def pack_jtypes(self):
        jtypes_f, jtypes_t = [], []
        for i, (j_f, j_t) in enumerate(self.skeleton):
            jtypes_f.append(j_f)
            jtypes_t.append(j_t)
        return jtypes_f, jtypes_t

    def _channel_dets(self, dets, jtypes):
        det_f = [temp[:, jtypes, :].unsqueeze(-1) for temp in dets]
        kps_scores_f, kps_inds_f, kps_ys_f, kps_xs_f = det_f
        kps_xys_f = torch.cat((kps_xs_f, kps_ys_f), dim=-1)
        return kps_inds_f, kps_scores_f, kps_xys_f

    def naive_generate_limbs(self):
        # shape of each item of dets: (N, 17, K), in which K equals self.topk
        dets = self.joint_dets()
        n, c, h, w = self.hmps_hr.shape
        n_limb = len(self.skeleton)

        # 将最外层循环矢量化, temp gather based on connections index
        for i, (jtype_f, jtype_t) in enumerate(self.skeleton):
            # shape of each returned tensors: (N, K, 1)
            det_f = [temp[:, jtype_f, :].unsqueeze(2) for temp in dets]
            kps_score_f, kps_inds_f, kps_ys_f, kps_xs_f = det_f  # 4 * (N, K, 1)
            kps_xys_f = torch.cat((kps_xs_f, kps_ys_f), dim=-1)  # (N, K, 2)

            det_t = [temp[:, jtype_t, :].unsqueeze(2) for temp in dets]
            kps_score_t, kps_inds_t, kps_ys_t, kps_xs_t = det_t  # 4 * (N, K, 1), can alose be (N, M, 1)
            kps_xys_t = torch.cat((kps_xs_t, kps_ys_t), dim=-1)  # (N, K, 2), can also be (N, M, 2)

            # ############### get offset vectors of limb connection i ###############
            offs_i = self.offs_hr.view((n, -1, 2, h, w))[:, i, ...]  # (N, 2, H, W)
            flat_off_i = offs_i.view((n, 2, -1))  # stretch and flat to (N, 2, H*W)
            kps_inds_f_expand = kps_inds_f.permute((0, 2, 1)).expand(n, 2, -1)  # (N, 2, K)
            kps_off_f = flat_off_i.gather(2, kps_inds_f_expand).permute((0, 2, 1))  # (N, 2, K) -> (N, K, 2)

            # ############### get the regressed end-joints from the start-joints ###############
            kps_guid_t = kps_xys_f + kps_off_f * self.resize_factor  # (N, K, 2)

            # ############### find limbs from kps_f_i to kps_t_j ###############
            kps_guid_t_expand = kps_guid_t.unsqueeze(2).expand(n, self.K, self.K, 2)  # (N, K, M, 2)
            kps_xys_t_expand = kps_xys_t.unsqueeze(1).expand(n, self.K, self.K, 2)  # (N, K, M, 2)
            dist = (kps_guid_t_expand - kps_xys_t_expand).norm(dim=3)  # (N, K, M)
            min_dist, min_ind = dist.min(dim=2)  # 2 * (N, K）
            min_dist = min_dist.unsqueeze(2)  # (N, K, 1)
            min_ind = min_ind.unsqueeze(2)  # (N, K, 1)

            matched_kps_score_t = kps_score_t.gather(1, min_ind)  # (N, K, 1)
            matched_kps_xys_t = kps_xys_t.gather(1, min_ind.expand(n, self.K, 2))  # (N, K, 2)

            # ###############  convert to global indexes across heatmap channels ###############
            matched_kps_inds_t = kps_inds_t.gather(1, min_ind) + jtype_t * h * w  # (N, K, 1)
            kps_inds_f = kps_inds_f + jtype_f * h * w  # (N, K, 1)

            # t1 = kps_guid_t.cpu().numpy()
            # t2 = kps_xys_t.cpu().numpy()
            # t3 = kps_xys_f.cpu().numpy()
            # y1 = kps_score_f.cpu().numpy()
            # tt1 = min_dist.cpu().numpy()
            # tt2 = min_ind.cpu().numpy()
            # t33 = matched_kps_inds_t.cpu().numpy()
            # t22 = matched_kps_score_t.cpu().numpy()
            # t11 = matched_kps_xys_t.cpu().numpy()
            pass


class GreedyGroup(object):
    """
    Greedily group the limbs into individual human skeletons in one image.
    """
    def __init__(self, limbs, scsmp, skeleton, threshold):
        self.limbs = limbs
        self.scsmp = scsmp
        self.skeleton = skeleton
        self.threshold = threshold

    def grop_skeletons(self, force_complete=False):
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20, 2))