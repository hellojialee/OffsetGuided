import copy
import logging
import numpy as np
import PIL
import warnings
import torchvision

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class CenterPad(Preprocess):
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def __call__(self, image, anns, meta, mask_miss=None):
        if mask_miss is not None:
            warnings.warn('mask_miss transformation is not implemented')
        image = PIL.Image.fromarray(image)
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]
        meta['width_height'] = np.array(image.size)
        LOG.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s', meta['valid_area'])
        image = np.asarray(image)
        return image, anns, meta, mask_miss

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
        anns[:, :, 0] += ltrb[0]
        anns[:, :, 1] += ltrb[1]

        return image, anns, ltrb


class SquarePad(Preprocess):
    def __call__(self, image, anns, meta, mask_miss):
        center_pad = CenterPad(max(image.shape[:2]))
        return center_pad(image, anns, meta, mask_miss)


class RightDownPad(Preprocess):
    def __init__(self, max_stride):
        """
        Args:
            max_stride: the max stride through the whole network
        """
        self.max_stride = max_stride

    def __call__(self, image, anns, meta, mask_miss=None):
        if mask_miss is not None:
            warnings.warn('mask_miss transformation is not implemented, '
                          'cannot be used during training')
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, anns, ltrb = self.corner_pad(image, anns)
        meta['offset'] -= ltrb[:2]
        meta['width_height'] = np.array([image.shape[1], image.shape[0]])
        LOG.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s', meta['valid_area'])

        return image, anns, meta, mask_miss

    def corner_pad(self, image, anns):
        fill = np.array([124, 116, 104], dtype=np.uint8).reshape((1, 1, 3))
        # if we use right-bottom pad, we don't have to consider keypoint offset
        h, w = image.shape[:2]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % self.max_stride == 0) else self.max_stride - (h % self.max_stride)  # down
        pad[3] = 0 if (w % self.max_stride == 0) else self.max_stride - (w % self.max_stride)  # right

        img_padded = image
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + fill, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        # notice: do not change the concatenation sequence
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + fill, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + fill, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + fill, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

        ltrb = (pad[1], pad[0], pad[3], pad[2])

        anns[:, :, 0] += ltrb[0]
        anns[:, :, 1] += ltrb[1]

        return img_padded, anns, ltrb
