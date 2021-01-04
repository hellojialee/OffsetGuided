import copy
import logging
import warnings
import numpy as np
import math
import cv2
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


def _scale(image, anns, meta, mask_miss, target_w, target_h, mode):
    """
    Args:
        anns (np.ndarray): shape of (N, 17, 4), the last dim includes x, y, v, scale

    target_w and target_h as integers
    """
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)

    # scale image
    h, w = image.shape[:2]
    image = cv2.resize(image, (target_w, target_h), interpolation=mode)
    LOG.debug('before resize = (%f, %f), after = %s', w, h, image.size)

    # mask_miss = cv2.resize(mask_miss, (target_w, target_h), interpolation=mode)  # not implemented
    # # mask_miss area marked by 0.
    # mask_miss = (mask_miss > 0.5).astype(np.uint8) * 255

    # rescale keypoint annotations
    # for example, 4 pixel cells (feature map cells) actually has the length of 3，
    # keypoints' coordinates are floating-point valuse，
    # I cannot tell where is the coordinate origin of keypoint annotation?
    x_scale = (target_w - 1) / (w - 1)  # (w-1)/(w'-1) not w/w'!
    y_scale = (target_h - 1) / (h - 1)
    # # refer to transforms.annotations.NormalizeAnnotations.normalize_annotations
    anns[:, :, 0] *= x_scale
    anns[:, :, 1] *= y_scale  # x_scale and y_scale may be different

    """
    to be checked :)
    heatmap和offset编码解码应该没有问题，但是坐标变换总是时常出现一个像素点偏移. 
    另外如果使用下面的变换，AP反而降低了0.3AP
    和transforms.preprocess.Preprocess.annotations_inverse中的变换保持一致，
    个人认为最重要的是保持编码（下采样生成featuremap)和解码(把预测的featuremap放大到原始图片尺度）的过程一致即可，
    使用哪种对齐方式可能影响不大，保险起见在随机变换原始图片和对应的标注时也应该统一对齐方式即可。
    我们也尝试测试时使用如下代码替换上面，但结果几乎完全一样 https://github.com/vita-epfl/openpifpaf/issues/183
    x_scale = target_w / w
    y_scale = target_h / h
    anns[:, :, 0] = (anns[:, :, 0] + 0.5) * x_scale - 0.5
    anns[:, :, 1] = (anns[:, :, 1] + 0.5) * y_scale - 0.5"""
    # x_scale = target_w / w
    # y_scale = target_h / h
    # anns[:, :, 0] = (anns[:, :, 0] + 0.5) * x_scale - 0.5
    # anns[:, :, 1] = (anns[:, :, 1] + 0.5) * y_scale - 0.5

    anns[:, :, 3] *= math.sqrt(x_scale * y_scale)  # resize the keypoint scale

    # adjust meta
    scale_factors = np.array((x_scale, y_scale))  # for w and h
    LOG.debug('meta before: %s', meta)
    meta['offset'] *= scale_factors
    meta['scale'] *= scale_factors
    meta['width_height'] = np.array([target_w, target_h])
    meta['valid_area'][:2] *= scale_factors  # valid_area的编码方式为(x1, y1, w, h)
    meta['valid_area'][2:] *= scale_factors
    LOG.debug('meta after: %s', meta)

    return image, anns, meta, mask_miss


class RescaleLongAbsolute(Preprocess):
    """ According to the given longest image edge to zoom out/in the image during testing"""
    def __init__(self, long_edge, *, resample=cv2.INTER_CUBIC):
        self.long_edge = long_edge
        self.mode = resample

    def __call__(self, image, anns, meta, mask_miss=None):
        if mask_miss is not None:
            warnings.warn('mask_miss transformation is not implemented,'
                          ' cannot be used during training')

        h, w = image.shape[:2]

        this_long_edge = self.long_edge
        if isinstance(this_long_edge, (tuple, list)):
            # This cloud raise a BUG during inference! You can not use tuple lengths during testing!
            this_long_edge = int(torch.randint(this_long_edge[0], this_long_edge[1], (1,)).item())

        s = this_long_edge / max(h, w)  # 缩放系数
        if h > w:
            target_w, target_h = int(w * s), this_long_edge
        else:
            target_w, target_h = this_long_edge, int(h * s)
        return _scale(image, anns, meta, mask_miss, target_w, target_h, self.mode)


class RescaleHighAbsolute(Preprocess):
    """According to the given image height to zoom out/in the image during testing"""
    def __init__(self, height_edge, *, resample=cv2.INTER_CUBIC):
        self.height_edge = height_edge
        self.mode = resample
        LOG.info(
            'you chose to rescale the the input image to the fixed height of %d',
            height_edge)

    def __call__(self, image, anns, meta, mask_miss=None):
        if mask_miss is not None:
            warnings.warn('mask_miss transformation is not implemented, '
                          'cannot be used during training')

        h, w = image.shape[:2]

        s = self.height_edge / h

        target_w, target_h = int(w * s), int(self.height_edge)

        return _scale(image, anns, meta, mask_miss, target_w, target_h, self.mode)


class RescaleRelative(Preprocess):
    def __init__(self, scale_factor=1.0, *, resample=cv2.INTER_CUBIC):
        self.scale_factor = scale_factor
        self.mode = resample

    def __call__(self, image, anns, meta, mask_miss=None):
        if mask_miss is not None:
            warnings.warn('mask_miss transformation is not implemented, '
                          'cannot be used during training')
        h, w = image.shape[:2]

        target_w, target_h = int(w * self.scale_factor), int(h * self.scale_factor)
        return _scale(image, anns, meta, mask_miss, target_w, target_h, self.mode)
