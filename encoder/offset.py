import numpy as np
import cv2
import math
import logging
import torch
from config.coco_data import COCO_PERSON_SKELETON, COCO_PERSON_SIGMAS

LOG = logging.getLogger(__name__)


class OffsetMaps(object):
    """Offset map between adjacent keypoints and scale map of keypoints.

    Args:
        input_size (int, list): the input image size of (w, h) or square length.

    Attributes:
        include_scale (bool): generate keypoint scale maps or not.
    """
    fill_scale_size = 7  # the diameter of the filled area
    # around keypoints will be filled with joint scale and adjacent-joint offset
    min_jscale = 1.0  # keypoint scale below this minimum are ignored # refer to transforms/annotations.py:27
    skeleton = COCO_PERSON_SKELETON  # human skeleton connections
    include_scale = True

    def __init__(self, input_size, stride):
        assert isinstance(input_size, (int, list)), input_size
        assert stride != 0, 'stride can not be zero'
        self.input_size = input_size if isinstance(input_size, list) \
            else [input_size] * 2
        self.in_out_scale = 1 / stride
        self.stride = stride

    def __call__(self, anns, meta, mask_miss):
        assert meta['width_height'][0] == self.input_size[0], 'raw data!'
        if not isinstance(self.stride, int):
            LOG.warning('network stride: %.3f is not a integer', self.stride)
            print(f'network stride: {self.stride: .3f} is not a integer')

        omps = OffsetMapGenerator(self.input_size, self.stride,
                                  self.fill_scale_size, self.min_jscale,
                                  self.skeleton)

        offset_maps, scale_maps, pscale_maps = omps.create_offsetmaps(anns, meta)

        mask_miss = cv2.resize(mask_miss, (0, 0),
                               fx=self.in_out_scale, fy=self.in_out_scale,
                               interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        # mask_miss area marked by 0.
        mask_miss = (mask_miss > 0.7)  # use bool instead of .astype(np.float32)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.repeat(mask_miss[:, :, np.newaxis], 3, axis=2))  # mask_all
        # plt.show()

        # Pytorch needs N*C*H*W format
        if self.include_scale:
            return (
                torch.from_numpy(offset_maps),
                torch.from_numpy(scale_maps),
                torch.from_numpy(pscale_maps), 
                torch.from_numpy(mask_miss[None, ...])
            )
        else:
            return (
                torch.from_numpy(offset_maps),
                torch.tensor([]),
                torch.from_numpy(pscale_maps),
                torch.from_numpy(mask_miss[None, ...]),
            )


class OffsetMapGenerator(object):
    """Generate guiding offset, keypoint scale and person scale feature map of keypoints.
    """

    def __init__(self, input_size, stride, fill_scale_size, min_jscale, skeleton):

        self.in_w = input_size[0]
        self.in_h = input_size[1]
        self.stride = stride
        # the area around keypoints will be filled with joint scale
        self.fill_scale_size = fill_scale_size
        # minimum keypoint scale
        self.min_jscale = min_jscale
        self.skeleton = skeleton

        # cached common parameters which same for all iterations and all pictures
        self.out_w = self.in_w // stride
        self.out_h = self.in_h // stride

        LOG.debug('Input image size: %d*%d, network output size: %d*%d',
                  self.in_w, self.in_h, self.out_w, self.out_h)

        # mapping coordinates into original input with cell center alignment.
        self.grid_x = np.arange(self.out_w) * stride + stride / 2 - 0.5  # x -> width
        self.grid_y = np.arange(self.out_h) * stride + stride / 2 - 0.5  # y -> height

    def create_offsetmaps(self, joints, meta):
        # print(joints.shape)  # e.g. (5, 17, 3)
        assert meta['joint_num'] == joints.shape[1], 'num of joints mismatch'

        channel_num = meta['joint_num']
        offset_num = len(self.skeleton)

        offset_maps = np.full((self.out_h, self.out_w, offset_num * 2), np.inf, dtype=np.float32)
        scale_maps = np.full((self.out_h, self.out_w, channel_num), np.nan, dtype=np.float32)
        pscale_maps = np.full((self.out_h, self.out_w, offset_num * 2), 1.0, dtype=np.float32)
        feature_maps = (offset_maps, scale_maps, pscale_maps)

        # ----------------------------------------------------------------------------------#
        # generate offset by sampling floating point positions in the original input resolution space
        self.put_connections(feature_maps, joints)

        return offset_maps.transpose((2, 0, 1)), scale_maps.transpose((2, 0, 1)), pscale_maps.transpose((2, 0, 1))

    def put_connections(self, feature_maps, joints):

        for limb_id, (fr, to) in enumerate(self.skeleton):
            visible_from = joints[:, fr, 2] > 0  # keypoint is annotated
            visible_to = joints[:, to, 2] > 0
            visible = visible_from & visible_to  # only annotated keypoints generate labels!

            self.put_guide_offsets(feature_maps, limb_id,
                                   joints[visible, fr, :], joints[visible, to, :],
                                   fr)

    def put_guide_offsets(self, feature_maps, limb_id, joints_fr, joints_to, fr):
        """
        Generate ground truth on a single channel.
        """

        # https://nedbatchelder.com/text/names.html
        # Values fall into two categories based on their type: mutable or immutable.
        # Immutable values include numbers, strings, and tuples. Almost everything
        # else is mutable, including lists, dicts, and user-defined objects. Mutable
        # means that **the value has methods that can change the value in-place**.
        # 翻译一下就是 mutable values可以in-place更改内存内容，
        # 而immutable values不能原地更改，实际上是新生成了value然后返回赋值
        # Immutable means that the value can never change, instead when you think you
        # are changing the value, you are REALLY making new values from old ones.

        # tuple 是不可改的，但是如果它的内部元素可改就另说了，
        # https://blog.csdn.net/lzw2016/article/details/85012814
        # 如果直接用tuple索引复制更改，虽然可以更改原始元素内存，但因为tuple不支持 = assign，所以仍然报错

        # In our case, we refer feature_maps[0] and feature_maps[1] as they are user-defined objects.
        # Thus, we can change the original storage in-place.
        offset_maps = feature_maps[0]
        scale_maps = feature_maps[1]
        pscale_maps = feature_maps[2]

        for joint1, joint2 in zip(joints_fr, joints_to):

            x_min = int(round(joint1[0] / self.stride - self.fill_scale_size / 2))
            x_max = int(round(joint1[0] / self.stride + self.fill_scale_size / 2))
            y_min = int(round(joint1[1] / self.stride - self.fill_scale_size / 2))
            y_max = int(round(joint1[1] / self.stride + self.fill_scale_size / 2))

            if y_max < 0:
                continue

            if x_max < 0:
                continue

            if x_min < 0:
                x_min = 0

            if y_min < 0:
                y_min = 0

            # this slice is not only to speed up (only compute the labels in needed areas,
            # but crop keypoints off the image boarder.
            # slice crops the extended index of a numpy array and return empty array []
            slice_x = slice(x_min, x_max)  # + 1
            slice_y = slice(y_min, y_max)  # + 1

            offset_x = (joint2[0] - self.grid_x[slice_x].astype(np.float32))  # type: np.ndarray
            # joint2[i, 1] -> y
            offset_y = (joint2[1] - self.grid_y[slice_y].astype(np.float32))
            offset_x_mesh = np.repeat(offset_x.reshape(1, -1), offset_y.shape[0], axis=0)
            offset_y_mesh = np.repeat(offset_y.reshape(-1, 1), offset_x.shape[0], axis=1)
            offset_mesh = np.concatenate(
                (offset_x_mesh[..., None], offset_y_mesh[..., None]),
                axis=-1)
            offset_mesh_l = np.linalg.norm(offset_mesh, axis=-1)

            offset_patch = offset_maps[slice_y, slice_x, 2 * limb_id: 2 * limb_id + 2]
            vector_l = np.linalg.norm(offset_patch, axis=-1)
            scale_patch = scale_maps[slice_y, slice_x, fr]
            pscale_patch = pscale_maps[slice_y, slice_x, 2 * limb_id: 2 * limb_id + 2]

            mask = offset_mesh_l < vector_l

            # overlap the offset values on the basis of the offset lengths
            offset_patch[mask] = offset_mesh[mask]
            scale_patch[mask] = joint1[3] if joint1[3] >= self.min_jscale else np.nan
            pscale_patch[mask] = joint1[3] / COCO_PERSON_SIGMAS[fr]

