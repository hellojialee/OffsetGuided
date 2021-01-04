"""
Borrowed from openpifpaf
"""
from abc import ABCMeta, abstractmethod
import copy
import numpy as np
from config.coco_data import RIGHT_INDEX, LEFT_INDEX


class Preprocess(metaclass=ABCMeta):

    def __call__(self, image, anns, meta, mask_miss):
        """Implementation of preprocess operation."""

    @staticmethod
    def affine_keypoint_inverse(keypoints, meta):
        """Inverse transform for WarpAffine augmentation in a single image"""
        keypoints = keypoints.copy()
        M = np.linalg.inv(meta['affine3×3mat'])
        original_joints = copy.deepcopy(keypoints)[..., :3]
        original_joints[:, :, 2] = 1  # Homogeneous coordinates
        # np.matmul regards the last two axis as matrix, and broadcast is used in our case.
        affine_joints = np.matmul(
            M[0:2],
            original_joints.transpose([0, 2, 1])).transpose([0, 2, 1])
        keypoints[:, :, 0:2] = affine_joints
        keypoints[:, :, 3] /= np.sqrt(np.prod(meta['scale']))  # keypoint scales
        # channel indexing.
        keypoints = keypoints[:, meta['joint_channel_ind'], :]
        return keypoints

    @staticmethod
    def annotations_inverse(keypoints, meta):  # has been checked, it's OK
        """Inversely transform all the person poses in the same scene to the original image space"""

        keypoints = copy.deepcopy(keypoints)

        # in the earlier data transformation time,
        # we firstly resize (scale) and then pad (shift) the image
        keypoints[:, :, 0] += meta['offset'][0]
        keypoints[:, :, 1] += meta['offset'][1]

        keypoints[:, :, 0] /= meta['scale'][0]
        keypoints[:, :, 1] /= meta['scale'][1]

        """
        This part should keep the same as in transforms.scale._scale! 
        It nearly makes no change in the inference
        keypoint_sets[:, :, 0] = (keypoint_sets[:, :, 0] + 0.5) / meta['scale'][0] - 0.5
        keypoint_sets[:, :, 1] = (keypoint_sets[:, :, 1] + 0.5) / meta['scale'][1] - 0.5"""
        # keypoints[:, :, 0] = (keypoints[:, :, 0] + 0.5) / meta['scale'][0] - 0.5
        # keypoints[:, :, 1] = (keypoints[:, :, 1] + 0.5) / meta['scale'][1] - 0.5

        keypoints[:, :, 3] /= np.sqrt(np.prod(meta['scale']))  # keypoint scales

        if meta['hflip']:
            w = meta['width_height'][0]  # keypoint的标注是浮点数，图像矩阵尺寸量化是整数，所以实际浮点数图像宽w-1
            keypoints[:, :, 0] = -keypoints[:, :, 0] + (w - 1)
            if meta.get('horizontal_swap'):
                keypoints[:] = meta['horizontal_swap'](keypoints)
            raise Exception('this should not happen. please have a check here, not implemented actually!')

        return keypoints
