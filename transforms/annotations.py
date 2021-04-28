import copy
import logging

import numpy as np
import math
import torch

from .preprocess import Preprocess
from config.coco_data import COCO_KEYPOINTS, COCO_PERSON_SIGMAS

LOG = logging.getLogger(__name__)


def scale_from_keypoints(keypoints):
    """The keypoint scale defined by the square root of the enveloping box of keypoints.
    ####  Copied from PIFPAF  ####
    """
    visible = keypoints[:, 2] > 0
    if not np.any(visible):
        raise Exception("no visiable keypoint in this instance")

    area = (
        (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
        (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
    )
    scale = np.sqrt(area)
    if scale < 0.01:
        scale = np.nan

    LOG.debug('instance scale = %.3f', scale)
    return scale


class NormalizeAnnotations(Preprocess):
    """
    Convert keypoint annotation into numpy array of shape (num_people, 17, 4)
    with (x, y, v, scale) at axis=-1. Initialize image meta info if dose not exist.
    """

    @staticmethod  # prepare the keypoint data
    def normalize_annotations(anns):
        anns = copy.deepcopy(anns)

        anns = [ann for ann in anns if ann['iscrowd'] == 0 and ann['num_keypoints'] > 0]
        num_people = len(anns)
        keypoints = np.zeros((num_people, len(COCO_KEYPOINTS), 4), dtype=np.float32)

        for i, ann in enumerate(anns):
            # notice the reshape's mechanism: wrap the every 3 element firstly.
            keypoints[i, :, :3] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            #  TODO: the root of areas? or instance bbox ? or the keypoints bbox?
            # scale = math.sqrt(ann['area'])  # root of areas
            # scale = scale_from_keypoints(keypoints[i, :, :3])  # root of instance bbox
            scale = math.sqrt(ann['bbox'][-1] * ann['bbox'][-2])  # root of keypoints bbox
            keypoints[i, :, 3] = scale * np.array(COCO_PERSON_SIGMAS)

            # actually small objects (segment area < 32^2) in COCO do not contain keypoint annotations,
            # thus you can just remove these two line codes
            if ann['area'] <= 32 * 32:
                keypoints[i, :, 2] = 0
        # print('ground truth \n', keypoints)

        return keypoints

    def __call__(self, image, anns, meta, mask_miss):
        anns = self.normalize_annotations(anns)

        if meta is None:
            h, w = image.shape[:2]
            # meta records the original image information, such as crop, scale, h, w
            meta = {
                'joint_num': len(COCO_KEYPOINTS),
                'offset': np.array([0.0, 0.0]),
                'scale': np.array([1.0, 1.0]),  # scale factors of width (x) and height (y)
                'valid_area': np.array([0.0, 0.0, w, h]),
                'hflip': False,
                'rotate': 0.,
                'width_height': np.array([w, h]),
                'original_width_height': np.array([w, h]),
                'affine3×3mat': np.array([[1., 0., 0],
                                          [0., 1., 0],
                                          [0., 0., 1.]], dtype=np.float32),
                'joint_channel_ind': np.arange(len(COCO_KEYPOINTS))
            }

        return image, anns, meta, mask_miss


class AnnotationJitter(Preprocess):
    """ Jitter the keypoint coordinates.

    Only used in 0 or 1 binary label, not proper for Gaussian regression.
    关键点标注添加抖动会是的预测的高亮相应范围变大，不是很好，比如造成断裂，一个人的身体出现两个半截pose
    但是，我们发现我们的groundtruth x,y常常会小1个像素（偏移），可以用annotaion增广抖动补偿
    """

    def __init__(self, shift=0, epsilon=0.5):
        self.shift = shift
        self.epsilon = epsilon

    def __call__(self, image, anns, meta, mask_miss):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        for ann in anns:  # loop over each person's annotation
            keypoints_xy = ann[:, :2]  # slice reference
            sym_rnd = (torch.rand(*keypoints_xy.shape).numpy() - 0.5 + self.shift) * 2.0
            # torch.rand生成0-1均匀分布，上面一步减去0.5，不加上1时，相当于-0.5~0.5均匀分布.
            keypoints_xy += self.epsilon * sym_rnd

        return image, anns, meta, mask_miss
