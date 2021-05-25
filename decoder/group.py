"""Greedily group the keypoints based on guiding (associative) offsets in ONE image"""
import logging
import time
import random
import functools
import numpy as np
from config.coco_data import (COCO_KEYPOINTS,
                              COCO_PERSON_SKELETON,
                              COCO_PERSON_WITH_REDUNDANT_SKELETON,
                              REDUNDANT_CONNECTIONS,
                              KINEMATIC_TREE_SKELETON)

LOG = logging.getLogger(__name__)


class GreedyGroup(object):
    """
    Greedily group the limbs into individual human skeletons in ONE image.

    Args:
        person_thre (float): threshold of pose instance scores to filter individual poses.
            If we sort them smaller, then we can choose a smaller value to detect more person poses.
        sort_dim (int): sort the person poses by the values at the this axis, which is used in _delete_sort.
            2th dim means keypoints score, 4th dim means limb score.
        dist_max (float): abandon limbs with delta offsets bigger than dist_max, if keypoint scales are unavailable.
        use_scale (bool): use the inferred keypoint scales.
    """

    def __init__(self, person_thre, *, sort_dim=2, dist_max=10, use_scale=False,
                 keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON):
        self.person_thre = person_thre
        self.use_scale = use_scale
        self.sort_dim = sort_dim
        self.skeleton = skeleton
        self.keypoints = keypoints
        self.dist_max = dist_max
        self.n_keypoints = len(keypoints)

    def group_skeletons(self, limbs):
        """
        Group all candidate limbs in a single image into individual human poses.
        This is not a batch-input operation!

        Args:
            limbs (np.ndarray): (L, K, 13), includes all limbs from the same image. The last dim includes:
              [x1, y1, v1, x2, y2, v2, ind1, ind2, len_delta (min_dist), len_limb, limb_score, scale1, scale2]
              0,    1, 2,  3,  4,  5,    6,   7,          8,                9,         10,        11,    12

        Returns:
            subset of keypoints (np.ndarray): shape [M * [ 17 * [x, y, v, s, limb_score, ind]]]
        """
        assert len(limbs) == len(self.skeleton
                                 ), 'check the skeleton config and input limbs Tensor'
        # subset shape is (0, 17, 6), mat be (M, 17, 6),
        # the last dim includes [x, y, v, s, limb_score, ind]
        start_time = time.time()
        subset = -1 * np.ones((0, self.n_keypoints, 6), dtype=np.float32)
        row_cache = self.subset_cache((limbs.shape[1], self.n_keypoints, 6))  # (K, 17, 6)

        # Loop over all kinds of Limb types
        for i, ((jtype_f, jtype_t), conns) in enumerate(zip(self.skeleton, limbs)):
            LOG.debug('limbs from jtype_f %d --> jtype_t %d', jtype_f, jtype_t)

            if self.use_scale:  # conns: (K, 13)   #  np.minimum((self.dist_max, conns[:, 12]) is bad
                dist_valid = conns[:, 8] < np.maximum(self.dist_max, conns[:, 12])  # 12: joint_t scale
            else:  # conns: (K, 10)
                dist_valid = conns[:, 8] < self.dist_max
                # conns = np.hstack((conns, 4 * np.ones((len(conns), 2))))  # we did this in LimbsCollect

            # ##########################################################################
            # ######################## remove false limbs ##############################
            # ##########################################################################
            # mask out the limbs with low responses which were moved off the image
            valid = dist_valid & (conns[:, 0] > 0) & (conns[:, 4] > 0) & (
                    conns[:, 3] > 0) & (conns[:, 1] > 0)  # we can ignore
            conns = conns[valid]  # (K, 11), may be (kk, 11) in which kk<K

            # ###########################################################################
            # ############ delete limb connections sharing the same keypoint ############
            # ###########################################################################
            conns = self._delete_reconns(conns)
            kk = len(conns)
            mm = len(subset)
            if kk == 0:
                continue

            jIDtab = subset[:, [jtype_f, jtype_t], -1]  # (M, 2)
            sub_scores = subset[:, [jtype_f, jtype_t], -2]  # (M, 2)

            xyvs1 = conns[:, [0, 1, 2, 11]]  # (K, 4)
            xyvs2 = conns[:, [3, 4, 5, 12]]  # (K, 4)
            limb_inds = conns[:, 6:8]  # (K, 2), joint_f_ID and joint_t_ID
            limb_scores = conns[:, [8, 10]]  # (K, 2), delta_length, limb_scores

            # suppose there are M pose skeletons, then the expanded shape is (M, K, 2)
            # apart from repeat, we can also use numpy array broadcast
            jIDtab_expand = np.expand_dims(jIDtab, axis=1)  # .repeat(kk, axis=1)  # (M, K, 2)
            sub_scores_expand = np.expand_dims(sub_scores, axis=1)  # .repeat(kk, axis=1)  # (M, K, 2)

            limb_inds_expand = np.expand_dims(limb_inds, axis=0)  # .repeat(mm, axis=0)  # (M, K, 2)
            limb_scores_expand = np.expand_dims(limb_scores, axis=0)  # .repeat(mm, axis=0)  # (M, K, 2)

            mask_sum = np.sum((jIDtab_expand.astype(int) == limb_inds_expand.astype(int)),
                              axis=-1)  # (M, K)
            # ########################################################################
            # ######## criterion to judge if we replace the exiting keypoints ########
            # ########################################################################
            replace_mask = (limb_scores_expand[..., 1] > sub_scores_expand[..., 1]) | (
                    limb_scores_expand[..., 1] > sub_scores_expand[..., 0])  # (M, K), score of joint_t

            # ########################################################################
            # ##### handle redundant limbs belonging to the same person skeleton #####
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
                # fixme: 有可能后来的冗余limb把一些点强行塞给了某些人，因为我们通过delete_conns只能保证endpoint只使用一次，
                #  不能保证它们又作为startpoint在之后的limb循环中被重复使用，是不是可以通过skeleton定义limb连接方向避免该问题
                subset[M_inds, jtype_f, -1] = limb_inds[K_inds, 0]
                subset[M_inds, jtype_t, -1] = limb_inds[K_inds, 1]
                subset[M_inds, jtype_f, :4] = xyvs1[K_inds]
                subset[M_inds, jtype_t, :4] = xyvs2[K_inds]
                # maybe the current limb shares the joint_f OR joint_t with some person skeleton
                subset[M_inds, jtype_f, 4] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_f, 4])
                subset[M_inds, jtype_t, 4] = np.maximum(limb_scores[K_inds, 1], subset[M_inds, jtype_t, 4])
                mask_sum[mask_sum == 1] = -1  # mask out the solved limbs

            # ########################################################################
            # ######## merge the subsets belonging to the same person skeleton #######
            # ########################################################################
            if mm >= 2:
                Msubset_expand = np.expand_dims(subset[..., -1],
                                                axis=1)  # .repeat(mm, axis=1)  # (M, M, 17) or (M, N, 17)
                Nsubset_expand = np.expand_dims(subset[..., -1],
                                                axis=0)  # .repeat(mm, axis=0)  # (M, M, 17) or (M, N, 17)
                merge_mask_sum = np.sum((Msubset_expand.astype(int) == Nsubset_expand.astype(int))
                                        & (Msubset_expand.astype(int) != -1),  # & (Nsubset_expand.astype(int) != -1)
                                        axis=-1)  # (M, M) or be (M, N)

                merge_mask_sum = np.triu(merge_mask_sum, 1)
                M_inds, N_inds = (merge_mask_sum == 2).nonzero()
                if len(M_inds):  # merge skeletons belonging to the same person
                    # overlay -1 elements or small scores, and keep the same keypoint info
                    subset[M_inds, :, :] = np.maximum(subset[M_inds, :, :], subset[N_inds, :, :])
                    # delete the merged old skeletons
                    subset = np.delete(subset, N_inds, axis=0)

                # other cases
                Mm_inds, Nm_inds = (merge_mask_sum >= 3).nonzero()
                if len(Mm_inds):  # todo: 取消分值小的所有连接，应该和mask_sum == 1情况下没处理好有关
                    print('usually this never happens, ignore handling skeletons crossing at 3 joints')
                    pass

            # ########################################################################
            # ########################## generate new skeletons ######################
            # ########################################################################
            New_inds, = (np.sum(mask_sum, axis=0) == 0).nonzero()  # sum(tensor of size[0])=0
            if len(New_inds):
                # rows = -1 * np.ones((len(New_inds), self.n_keypoints, 6))
                rows = row_cache[:len(New_inds)]  # slice: a view of the original storage
                rows[:, [jtype_f, jtype_t], -1] = limb_inds[New_inds]
                rows[:, jtype_f, :4] = xyvs1[New_inds]
                rows[:, jtype_t, :4] = xyvs2[New_inds]
                # initial two connected keypoint share the same limb_score
                rows[:, jtype_f, 4] = limb_scores[New_inds, 1]
                rows[:, jtype_t, 4] = limb_scores[New_inds, 1]
                subset = np.concatenate((subset, rows), axis=0)
                rows[:] = -1  # reset cache

        LOG.debug('keypoint grouping time in the current image: %.6fs', time.time() - start_time)

        subset = self._delete_sort(subset, self.person_thre, self.sort_dim)

        # subset = soft_nms(subset)  # make no difference owning to our Gaussian spread

        return subset  # numpy array [M * [ 17 * [x, y, v, s, limb_score, ind]]]

    @staticmethod
    def _delete_sort(subset, thre, index):  # todo: how about index=4? use limb_score to sort
        """
        Delete and sort the detected poses according to scores.
        Copied from openpifpaf (CVPR 2019).

        Args:
            subset (list): detected results of shape [M * [ 17 * [x, y, v, s, limb_score, ind]]]
                in a given image
            thre (float): threshold to filter the poses with low scores
            index (int): sort the person poses by the values at the this axis.
                2th dim means keypoints score, 4th dim means limb score

        Returns:
            subset (list): detected poses in a single image

        """
        t0 = time.time()
        delete_list = []
        scores_list = []
        for i in range(len(subset)):
            kps_mask = subset[i, :, index] > 0
            score = subset[i, kps_mask, index].sum() / kps_mask.astype(float).sum()
            if score < thre:
                delete_list.append(i)
            else:
                scores_list.append(score)
        subset = np.delete(subset, delete_list, axis=0)
        sort_inds = sorted(range(len(scores_list)), key=lambda k: scores_list[k], reverse=True)
        subset = subset[sort_inds]
        subset[subset == -1] = 0
        LOG.debug('delete and sort the poses: %.6fs', time.time() - t0)
        return subset

    @staticmethod
    def _delete_reconns(conns):
        """
        Sort and delete reused keypoint, ensuring one keypoint can only be used once
        by adjacent keypoints in the skeleton.

        Args:
            conns (np.ndarray): shape (K, 11), in which the last dim includes:
                [x1, y1, v1, x2, y2, v2, ind1, ind2, len_delta, len_limb, limb_score, scale1, scale2]
                # 0,  1, 2,  3,  4,  5,    6,   7,      8,         9,        10,        11,    12
        """
        conns = conns[np.argsort(-conns[:, 10])]  # 10: sort by limb_scores
        repeat_check = []
        unique_list = []
        for j, ind_t in enumerate(conns[:, 7].astype(int)):
            if ind_t not in repeat_check:
                repeat_check.append(ind_t)
                unique_list.append(j)
        conns = conns[unique_list]
        return conns

    @staticmethod
    @functools.lru_cache(maxsize=16)
    def subset_cache(shape):
        row_cache = - 1 * np.ones(shape, dtype=np.float32)
        return row_cache


def soft_nms(subset, suppressed_v=0):
    if not len(subset):
        return subset

    occupied = np.zeros((
        len(subset[0]),
        int(max(np.max(ann[:, 1]) for ann in subset) + 1),
        int(max(np.max(ann[:, 0]) for ann in subset) + 1),
    ), dtype=np.uint8)

    for ann in subset:
        joint_scales = np.maximum(10.0, ann[:, 3])

        assert len(occupied) == len(ann)
        for xyv, occ, joint_s in zip(ann[:, :3], occupied, joint_scales):
            v = xyv[2]
            if v == -1:
                continue
            # Use cython to speed up if needed：
            # https://python3-cookbook.readthedocs.io/zh_CN/latest/c15/p11_use_cython_to_write_high_performance_array_operation.html
            x = np.clip(xyv[0], 0.0, occ.shape[1] - 1).astype(int)
            y = np.clip(xyv[1], 0.0, occ.shape[0] - 1).astype(int)
            if occ[y, x]:
                xyv[2] = suppressed_v
            else:
                scalar_square_add_single(occ, xyv[0], xyv[1], joint_s, 1)
    return subset


def scalar_square_add_single(field, x, y, width, value):
    minx = max(0, int(x - width))
    miny = max(0, int(y - width))
    maxx = max(minx + 1, min(field.shape[1], int(x + width) + 1))
    maxy = max(miny + 1, min(field.shape[0], int(y + width) + 1))
    field[miny:maxy, minx:maxx] += value
