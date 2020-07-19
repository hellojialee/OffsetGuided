import copy
import logging
import numpy as np

import torchvision

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class CenterPad(Preprocess):  # FIXME: 用right-bottom pad我认为更好，这样不用对keypoint offset做更改了
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]

        LOG.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s', meta['valid_area'])

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    def center_pad(self, image, anns):
        w, h = image.size

        left = int((self.target_size[0] - w) / 2.0)
        top = int((self.target_size[1] - h) / 2.0)
        if left < 0:
            left = 0
        if top < 0:
            top = 0

        right = self.target_size[0] - w - left
        bottom = self.target_size[1] - h - top
        if right < 0:
            right = 0
        if bottom < 0:
            bottom = 0
        # 给出左，上，右，下分别的pad长度，后面torchvision.transforms.functional.pad将会识别出给了4个pad参数，从而开始pad
        ltrb = (left, top, right, bottom)

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(124, 116, 104))

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]  # x坐标加上图像左侧填充的像素个数
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, anns, ltrb


class SquarePad(Preprocess):
    def __call__(self, image, anns, meta):
        center_pad = CenterPad(max(image.size))
        return center_pad(image, anns, meta)


def padRightDownCorner(img, stride, padValue=128):  # padValue= (124, 116, 104)
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))  # img_padded[0:1, :] * 0 + padValue  # padValue= (124, 116, 104)
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    # 注意! concatenate 两个数组的顺序很重要
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad