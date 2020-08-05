from abc import ABCMeta, abstractmethod
import copy
import numpy as np


class Preprocess(metaclass=ABCMeta):

    def __call__(self, image, anns, meta, mask_miss):
        """Implementation of preprocess operation."""

    @staticmethod
    def affine_keypoint_inverse(anns, meta):  # to be tested and checked
        M = np.linalg.inv(meta['affine3×3mat'])
        original_joints = copy.deepcopy(anns)[..., :3]
        original_joints[:, :, 2] = 1  # Homogeneous coordinates
        # np.matmul regards the last two axis as matrix, and broadcast is used in our case.
        affine_joints = np.matmul(
            M[0:2],
            original_joints.transpose([0, 2, 1])).transpose([0, 2, 1])
        anns[:, :, 0:2] = affine_joints
        # channel indexing.
        anns[:, meta['joint_channel_ind'], :] = anns
        return anns