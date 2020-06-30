import copy
import logging

import numpy as np
import PIL
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


def _scale(image, anns, meta, target_w, target_h, resample):
    """target_w and target_h as integers"""
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)

    # scale image
    w, h = image.size  # 原始图像尺寸，image默认是PIL.Image类型，因此直接调用缩放属性，并且和cv2.resize一样使用中心点对齐了
    image = image.resize((target_w, target_h), resample)  # OpenCV默认插值用双线性，PIL默认用最临近插值
    LOG.debug('before resize = (%f, %f), after = %s', w, h, image.size)

    # rescale keypoints
    # 比如4个特征图单元(feature map cells)其实只相距为3个单位，这样坐标值变成了(w-1)/(w'-1)倍而不是w/w'
    # 这是因为keypoint标注的坐标是浮点数，keypoint是以图像坐标原点像素的最左上角为原点。另外评测带小数能明显提高AP
    x_scale = (image.size[0] - 1) / (w - 1)
    y_scale = (image.size[1] - 1) / (h - 1)
    for ann in anns:  # anns是所有人的ground truth信息
        ann['keypoints'][:, 0] = ann['keypoints'][:, 0] * x_scale
        ann['keypoints'][:, 1] = ann['keypoints'][:, 1] * y_scale
        ann['bbox'][0] *= x_scale  # bbox的编码形式为(x1, y1, x2, y2)
        ann['bbox'][1] *= y_scale
        ann['bbox'][2] *= x_scale
        ann['bbox'][3] *= y_scale

    """ 
    通过试验发现，即使最后关键点坐标用floor取整的情况下，上面这种缩放方式和下面这种(在version 0.5中采用)测试后的
    COCO AP指标几乎完全一样，看不出区别
    和transforms.preprocess.Preprocess.annotations_inverse中的变换保持一致，
    个人认为最重要的是保持编码解码的过程一致即可，使用哪种对齐方式可能影响不大，
    我们也尝试测试时使用如下代码替换上面，但结果几乎完全一样 https://github.com/vita-epfl/openpifpaf/issues/183
    x_scale = image.size[0] / w
    y_scale = image.size[1] / h
    for ann in anns:
        ann['keypoints'][:, 0] = (ann['keypoints'][:, 0] + 0.5) * x_scale - 0.5
        ann['keypoints'][:, 1] = (ann['keypoints'][:, 1] + 0.5) * y_scale - 0.5
    """
    # adjust meta
    scale_factors = np.array((x_scale, y_scale))
    LOG.debug('meta before: %s', meta)
    meta['offset'] *= scale_factors
    meta['scale'] *= scale_factors
    meta['valid_area'][:2] *= scale_factors  # valid_area的编码方式为(x1, y1, w, h)
    meta['valid_area'][2:] *= scale_factors
    LOG.debug('meta after: %s', meta)

    for ann in anns:
        ann['valid_area'] = meta['valid_area']

    return image, anns, meta


class RescaleRelative(Preprocess):  # 相对于自身尺寸进行缩放，用于训练时的数据增强
    def __init__(self, scale_range=(0.5, 1.0), *,
                 resample=PIL.Image.BICUBIC,
                 power_law=False):
        self.scale_range = scale_range
        self.resample = resample
        self.power_law = power_law

    def __call__(self, image, anns, meta):
        if isinstance(self.scale_range, tuple):
            if self.power_law:
                rnd_range = np.log2(self.scale_range[0]), np.log2(self.scale_range[1])
                log2_scale_factor = (
                    rnd_range[0] +
                    torch.rand(1).item() * (rnd_range[1] - rnd_range[0])
                )  # torch.rand: uniform distribution on the interval [0,1)
                # mean = 0.5 * (rnd_range[0] + rnd_range[1])
                # sigma = 0.5 * (rnd_range[1] - rnd_range[0])
                # log2_scale_factor = mean + sigma * torch.randn(1).item()

                scale_factor = 2 ** log2_scale_factor
                # LOG.debug('mean = %f, sigma = %f, log2r = %f, scale = %f',
                #           mean, sigma, log2_scale_factor, scale_factor)
                LOG.debug('rnd range = %s, log2_scale_Factor = %f, scale factor = %f',
                          rnd_range, log2_scale_factor, scale_factor)
            else:
                scale_factor = (
                    self.scale_range[0] +
                    torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
                )
        else:
            scale_factor = self.scale_range

        w, h = image.size
        target_w, target_h = int(w * scale_factor), int(h * scale_factor)
        return _scale(image, anns, meta, target_w, target_h, self.resample)


class RescaleAbsolute(Preprocess):  # 根据给出的具体尺寸进行缩放，用于测试时调整图像尺寸
    def __init__(self, long_edge, *, resample=PIL.Image.BICUBIC):
        self.long_edge = long_edge
        self.resample = resample

    def __call__(self, image, anns, meta):
        w, h = image.size

        this_long_edge = self.long_edge
        if isinstance(this_long_edge, (tuple, list)):
            this_long_edge = int(torch.randint(this_long_edge[0], this_long_edge[1], (1,)).item())

        s = this_long_edge / max(h, w)  # 缩放系数
        if h > w:
            target_w, target_h = int(w * s), this_long_edge
        else:
            target_w, target_h = this_long_edge, int(h * s)
        return _scale(image, anns, meta, target_w, target_h, self.resample)
