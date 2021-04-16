"""Efficient WarpAffine transforms for images and labels during our training process"""
from .preprocess import Preprocess
import numpy as np
import random
import math
import cv2
import copy
import logging
from config.coco_data import RIGHT_INDEX, LEFT_INDEX

LOG = logging.getLogger(__name__)


def _roi_center(anns, meta):
    """
    The center (x, y) of the area filled with keypoints.
    """
    if not len(anns):
        return meta['width_height'].astype(np.float32) // 2

    min_x = np.min(anns[anns[:, :, 2] > 0, 0])
    min_y = np.min(anns[anns[:, :, 2] > 0, 1])
    max_x = np.max(anns[anns[:, :, 2] > 0, 0])
    max_y = np.max(anns[anns[:, :, 2] > 0, 1])
    return np.array([(min_x + max_x) // 2, (min_y + max_y) // 2]).astype(np.float32)


class FixedAugParams(object):
    """
    An example of none-random augmentation params for warp affine.
    """
    def __init__(self):
        self._flip_prob = 0
        self._max_rotate = 0
        self._min_scale = 1.
        self._max_scale = 1.
        self._max_translate = 0.
        self._min_stretch = 1.
        self._max_stretch = 1.

    @property
    def flip_prob(self):
        """Get the current voltage."""
        return self._flip_prob

    @property
    def max_rotate(self):
        return self._max_rotate

    @property
    def min_scale(self):
        return self._min_scale

    @property
    def max_scale(self):
        return self._max_scale

    @property
    def min_stretch(self):
        return self._min_stretch

    @property
    def max_stretch(self):
        return self._max_stretch

    @property
    def max_translate(self):
        return self._max_translate


class WarpAffineTransforms(Preprocess):
    """
    We use OpenCV WarpAffine method to efficiently transform the images,
    labels, and keypoints during training.

    Resize, crop, shift, flip the image and labels randomly.

    Args:
        dst_size (int, list): the dst input image size of (width, height) or square length.
        aug_params (Namespace, Class, None): augmentation params used in data transformations.
        crop_roi (bool): crop the valid area including person keypoints
    """

    def __init__(self, dst_size, *,
                 aug_params=None, crop_roi=True, debug_show=False):
        assert isinstance(dst_size, (int, list)), dst_size
        self.in_size = dst_size if isinstance(dst_size, list) else [dst_size] * 2
        self.flip_prob = aug_params.flip_prob
        self.max_rotate = aug_params.max_rotate
        self.min_scale = aug_params.min_scale
        self.max_scale = aug_params.max_scale
        self.min_stretch = aug_params.min_stretch
        self.max_stretch = aug_params.max_stretch
        self.max_translate = aug_params.max_translate

        self.flip = False
        self.rotate = 0.
        self.scale = 1.
        self.x_offset = 0
        self.y_offset = 0
        self.x_stretch = 1
        self.y_stretch = 1

        self.crop_roi = crop_roi
        self.debug_show = debug_show

    def __call__(self, image, anns, meta, mask_miss):
        """
        Transform images, anns, meta and mask_miss which will be feed in to the network.
        Note that mask_miss keeps the same with the output size of network,
        while other data is in the original input resolution space.
        """
        # get random transformation params
        self.flip = random.uniform(0., 1.) < self.flip_prob
        self.rotate = random.uniform(-1., 1.) * self.max_rotate

        self.scale = (self.max_scale - self.min_scale
                      ) * random.uniform(0., 1.) + self.min_scale

        self.x_stretch = (self.max_stretch - self.min_stretch
                          ) * random.uniform(0., 1.) + self.min_stretch
        self.y_stretch = (self.max_stretch - self.min_stretch

                          ) * random.uniform(0., 1.) + self.min_stretch
        self.x_offset = int(random.uniform(-1., 1.) * self.max_translate)
        self.y_offset = int(random.uniform(-1., 1.) * self.max_translate)

        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        roi_center = _roi_center(anns, meta)
        # compute the transform matrix for image and keypoints
        affine_mat = self._get_affine_mat(roi_center, meta)
        M = affine_mat[0:2]

        # fill mean RGB values  255* array([0.485, 0.456, 0.406]), and
        # during training, torchvision.transforms.transforms.Normalize will subtract to zero.
        image = cv2.warpAffine(image, M,
                               (self.in_size[1], self.in_size[0]),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(124, 116, 104))
        if mask_miss is not None:
            mask_miss = self._affine_mask_miss(M, mask_miss)

        self._affine_keypoints(M, anns, meta)  # in-place transform

        if self.debug_show:
            self._show_affine_result(anns, image, mask_miss)

        meta['hflip'] = self.flip
        meta['scale'] *= np.array([self.scale_x, self.scale_y])
        meta['rotate'] += self.rotate
        meta['affine3×3mat'] = affine_mat.dot(meta['affine3×3mat'])
        meta['width_height'] = np.array(self.in_size)
        LOG.debug(meta)

        return image, anns, meta, mask_miss

    @staticmethod
    def _show_affine_result(anns, image, mask_miss):
        import matplotlib.pyplot as plt

        for i, xyvs in enumerate(anns[anns[:, :, 2] > 0]):
            if xyvs[3] > 0:
                cv2.circle(image, (int(xyvs[0]), int(xyvs[1])),
                       int(xyvs[3]), color=[126, 0, 100], thickness=2)

        plt.imshow(image)
        plt.show()
        LOG.debug('the shape of input image %d width * %d height',
                  image.shape[1], image.shape[0])
        plt.imshow(np.repeat(mask_miss[:, :, np.newaxis], 3, axis=2))
        plt.show()
        LOG.debug('transformed annotations %d, max_x %.3f, max_y %.3f',
                  len(anns), anns[:, :, 0].max(), anns[:, :, 1].max())

    def _affine_mask_miss(self, M, mask_miss):
        mask_miss = cv2.warpAffine(mask_miss, M,
                                   (self.in_size[1], self.in_size[0]),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)  # mask_miss area marked by 0
        # we will resize the mask_miss in ground truth encoder
        # mask_miss = cv2.resize(mask_miss, (0, 0),
        #                        fx=self.in_out_scale, fy=self.in_out_scale,
        #                        interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        # mask_miss area marked by 0.
        # mask_miss = (mask_miss > 0.5).astype(np.float32)
        return mask_miss

    def _affine_keypoints(self, M, anns, meta):
        """Affine he keypoint annotations in-place"""
        original_joints = copy.deepcopy(anns)[..., :3]
        # we reuse 3rd column, it is a trick.
        original_joints[:, :, 2] = 1
        # np.matmul regards the last two axis as matrix, and broadcast is used in our case.
        affine_joints = np.matmul(
            M, original_joints.transpose([0, 2, 1])).transpose([0, 2, 1])
        # 矩阵相乘的方式: 第一次transpose后坐标表示用的是列向量，所以是左乘变换矩阵
        anns[:, :, 0:2] = affine_joints

        # we must flip the keypoint channels accordingly.
        if self.flip:
            assert len(RIGHT_INDEX) == len(LEFT_INDEX), 'check left and right body keypoints'
            tmp_left = anns[:, LEFT_INDEX, :]
            tmp_right = anns[:, RIGHT_INDEX, :]
            # swap the left and right
            anns[:, LEFT_INDEX, :] = tmp_right
            anns[:, RIGHT_INDEX, :] = tmp_left
            LOG.debug('flip left and right keypoints during augmentation')
            meta['joint_channel_ind'][LEFT_INDEX] = RIGHT_INDEX
            meta['joint_channel_ind'][RIGHT_INDEX] = LEFT_INDEX

        # crop the keypoints beyond the image boarder
        for i, p in enumerate(anns):
            for j, xyvs in enumerate(p):

                anns[i, j, 3] *= math.sqrt(self.scale_x * self.scale_y)  # rescale the keypoint size
                # print(anns[i, j, 3])  # keypoint scales mostly vary form 0.8 to 22.

                if xyvs[0] <= 0 or xyvs[1] <= 0 \
                        or xyvs[0] > self.in_size[0] \
                        or xyvs[1] > self.in_size[1]:
                    anns[i, j, 2] = 0

        return self

    def _get_affine_mat(self, roi_center, meta):
        """
        Do all image transformations with one affine matrix.
        Same affine matrix could be used to transform joint coordinates afterwards.
        """
        cangle = math.cos(self.rotate / 180. * math.pi)
        sangle = math.sin(self.rotate / 180. * math.pi)

        self.scale_x = self.x_stretch * self.scale
        self.scale_y = self.y_stretch * self.scale

        # coordinates start from 0 to w-1 or h-1.
        (center_x, center_y) = (meta['width_height'] - 1).astype(np.float32) / 2
        (move2roi_x, move2roi_y) = np.array((center_x, center_y)) - roi_center
        translate_x = self.x_offset + (move2roi_x * self.scale_x if self.crop_roi else 0)
        translate_y = self.y_offset + (move2roi_y * self.scale_y if self.crop_roi else 0)

        # shift the image center to the Origin of the coordinate system for convenient.
        center2zero = np.array([[1., 0., -center_x],
                                [0., 1., -center_y],
                                [0., 0., 1.]])

        rotate = np.array([[cangle, sangle, 0],
                           [-sangle, cangle, 0],
                           [0, 0, 1.]])

        scale = np.array([[self.scale_x, 0, 0],
                          [0, self.scale_y, 0],
                          [0, 0, 1.]])

        flip = np.array([[-1 if self.flip else 1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])

        # shift back to the center of dst image.
        zero2center = np.array([[1., 0., (self.in_size[0] - 1) / 2],
                                [0., 1., (self.in_size[1] - 1) / 2],
                                [0., 0., 1.]])

        # random translation + crop
        center2center = np.array([[1., 0., translate_x],
                                  [0., 1., translate_y],
                                  [0., 0., 1.]])

        # order of combination is reversed because we use row vector [x, y, 1]
        # 这取决于坐标是行向量还是列向量，对应变换矩阵是左乘还是右乘，后面变换时坐标转成了列向量
        affine_mat = center2center.dot(zero2center).dot(flip).\
            dot(scale).dot(rotate).dot(center2zero)

        return affine_mat


