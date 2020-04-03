import copy
import logging

import numpy as np
import PIL

from .preprocess import Preprocess
from config.coco_data import COCO_KEYPOINTS, HFLIP

LOG = logging.getLogger(__name__)


class HorizontalSwap():
    def __init__(self, joints=None, hflip=None):
        self.joints = joints or COCO_KEYPOINTS
        self.hflip = hflip or HFLIP

    def __call__(self, keypoints):
        target = np.zeros(keypoints.shape)

        for source_i, xyv in enumerate(keypoints):
            source_name = self.joints[source_i]
            target_name = self.hflip.get(source_name)
            if target_name:
                target_i = self.joints.index(target_name)
            else:
                target_i = source_i
            target[target_i] = xyv

        return target


class HFlip(Preprocess):
    def __init__(self, *, swap=None):
        self.swap = swap or HorizontalSwap()  # 若swap=None，则默认返回HorizontalSwap()

    def __call__(self, image, anns, meta, mask_miss):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        _, w = image.shape[:2]
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        for ann in anns:  # todo: wait to check here. We do not know the COCO annotation coordinate references
            ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
            if self.swap is not None and not ann['iscrowd']:
                ann['keypoints'] = self.swap(ann['keypoints'])
                meta['horizontal_swap'] = self.swap
            ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

        assert meta['hflip'] is False
        meta['hflip'] = True

        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + w
        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta
