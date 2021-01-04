import numpy as np
import cv2
import math
import logging
import torch
from config.coco_data import COCO_KEYPOINTS

LOG = logging.getLogger(__name__)


class HeatMaps(object):
    """
    Gaussian heatmap of keypoints.

    Args:
        input_size (int, list): the input image size of (w, h) or square length.
        include_background (bool): include the reversed background heatmap or not
    """
    clip_thre = 0.01  # Gaussian distribution below this value will be set to zero
    sigma = 7  # a little better thant 9; standard deviation of Gaussian distribution
    n_keypoints = 17
    keypoints = COCO_KEYPOINTS
    include_background = True  # background heatmap
    include_jitter_offset = True  # jitter offsetmaps
    fill_jitter_size = 3  # the diameter of the refinement offset area to the nearest keypoint

    def __init__(self, input_size, stride):
        assert isinstance(input_size, (int, list)), input_size
        assert stride != 0, 'stride can not be zero'
        self.input_size = input_size if isinstance(input_size, list) \
            else [input_size] * 2
        # ratio of output length to input length, used by mask_miss
        self.in_out_scale = 1 / stride
        self.stride = stride

    def __call__(self, anns, meta, mask_miss):
        assert meta['width_height'][0] == self.input_size[0], 'raw data!'
        if not isinstance(self.stride, int):
            LOG.warning('network stride: %.3f is not a integer', self.stride)

        assert self.n_keypoints == meta['joint_num'], \
            'not implemented! n_keypoints set by command parse args mismatches the COCO config '

        # speed dose ont change even if we initialize the HeatMapGenerator in __init__()
        hmps = HeatMapGenerator(self.input_size, self.stride,
                                self.fill_jitter_size,
                                self.sigma, self.clip_thre)

        heatmaps = hmps.create_heatmaps(anns, meta)

        if self.include_jitter_offset:
            jittermaps = hmps.create_jitter_offset(anns, meta)
        else:
            jittermaps = np.array([], dtype=np.float32)

        mask_miss = cv2.resize(mask_miss, (0, 0),
                               fx=self.in_out_scale, fy=self.in_out_scale,
                               interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        # mask_miss area marked by 0.
        mask_miss = (mask_miss > 0.7)  # .astype(np.float32)  bool mask_miss

        # # for IDE debug only
        # import matplotlib.pyplot as plt
        # resize_hmps = cv2.resize(heatmaps, (0, 0),
        #                        fx=self.stride, fy=self.stride,
        #                        interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # plt.imshow(np.repeat(mask_miss[:, :, np.newaxis], 3, axis=2))  # mask_all
        # plt.show()
        # plt.imshow(resize_hmps[:, :, 1])
        # plt.show()

        LOG.debug('the shape of output mask_miss %d width * %d height',
                  mask_miss.shape[1], mask_miss.shape[0])

        # Pytorch needs N*C*H*W format
        if self.include_background:
            # add reverse keypoint gaussian heatmap as the background channel
            hmp_reverse = 1 - np.amax(heatmaps, axis=2)

            return (
                torch.from_numpy(heatmaps),
                torch.from_numpy(hmp_reverse[None, ...]),
                torch.from_numpy(jittermaps),
                torch.from_numpy(mask_miss[None, ...])
            )
        else:
            return (
                torch.from_numpy(heatmaps),
                torch.tensor([]),
                torch.from_numpy(jittermaps),
                torch.from_numpy(mask_miss[None, ...])
            )


class HeatMapGenerator(object):
    """
    Generate the keypoint heatmaps and keypoint scale feature map.
    """

    def __init__(self, input_size, stride, fill_jitter_size, sigma, clip_thre):

        self.in_w = input_size[0]
        self.in_h = input_size[1]
        self.stride = stride
        self.fill_jitter_size = fill_jitter_size
        self.sigma = sigma
        self.double_sigma2 = 2 * self.sigma * self.sigma
        # set responses lower than gaussian threshold to 0.
        self.gaussian_clip_thre = clip_thre
        # get the gaussian peak spread
        self.gaussian_size = 2 * math.ceil(
            (math.sqrt(-self.double_sigma2 * math.log(self.gaussian_clip_thre))) / stride)
        assert self.gaussian_clip_thre > 0 and self.gaussian_size > 0, 'should bigger than zero'
        # cached common parameters which same for all iterations and all pictures
        self.out_w = self.in_w // stride
        self.out_h = self.in_h // stride

        LOG.debug('Input image size: %d*%d, network output size: %d*%d',
                  self.in_w, self.in_h, self.out_w, self.out_h)

        # mapping coordinates into original input with cell center alignment.
        self.grid_x = np.arange(self.out_w) * stride + stride / 2 - 0.5  # x -> width
        self.grid_y = np.arange(self.out_h) * stride + stride / 2 - 0.5  # y -> height

    def create_heatmaps(self, joints, meta):
        """
        Create keypoint Gaussian heatmaps.
        """
        # print(joints.shape)  # e.g. (5, 17, 3)
        assert meta['joint_num'] == joints.shape[1], 'num of joints mismatch'
        channel_num = meta['joint_num']
        heatmaps = np.zeros((self.out_h, self.out_w, channel_num), dtype=np.float32)

        # ----------------------------------------------------------------------------------#
        # generate Gaussian peaks by sampling in the original input resolution space
        self.put_joints(heatmaps, joints, channel_num)

        return heatmaps.transpose((2, 0, 1))

    def put_joints(self, heatmaps, joints, channel_num):

        for i in range(channel_num):
            # loop over each keypoint channel
            # only annotated keypoints are considered !
            visible = joints[:, i, 2] > 0
            self.put_gaussian_peaks(heatmaps, i, joints[visible, i])

    def put_gaussian_peaks(self, heatmaps, layer, joints):
        """
        Generate ground-truth Gaussian responses on a single channel.
        """
        for i in range(joints.shape[0]):

            x_min = int(round(joints[i, 0] / self.stride - self.gaussian_size / 2))
            x_max = int(round(joints[i, 0] / self.stride + self.gaussian_size / 2))
            y_min = int(round(joints[i, 1] / self.stride - self.gaussian_size / 2))
            y_max = int(round(joints[i, 1] / self.stride + self.gaussian_size / 2))

            if y_max < 0:
                continue

            if x_max < 0:
                continue

            if x_min < 0:
                x_min = 0

            if y_min < 0:
                y_min = 0

            # this slice is not only speed up but crops the keypoints off the image boarder
            # slice can also crop the extended index of a numpy array and return empty array []
            # max_sx + 1: array slice dose not include the last element.
            slice_x = slice(x_min, x_max)  # + 1
            slice_y = slice(y_min, y_max)

            disk_x = self.grid_x[slice_x].astype(np.float32) - joints[i, 0]
            disk_y = self.grid_y[slice_y].astype(np.float32) - joints[i, 1]
            exp_x = np.exp(- disk_x ** 2 / np.array([self.double_sigma2]).astype(np.float32))
            exp_y = np.exp(- disk_y ** 2 / np.array([self.double_sigma2]).astype(np.float32))

            # np.outer of two vector with length M and N gets a matrix with shape M*N
            exp = np.outer(exp_y, exp_x)

            # # Laplace heatmap
            # dis = exp-(math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma)
            # dist = np.sqrt((self.X - joints[i, 0])**2 +
            #                (self.Y - joints[i, 1])**2) / 2.0 / self.sigma
            # dist = np.where(dist > 4.6052, 1e8, dist)
            # exp = np.exp(-dist)

            # overlap Gaussian peaks by taking the maximum
            # Must use slice view to overlap original array!
            exp[exp < self.gaussian_clip_thre] = 0
            patch = heatmaps[slice_y, slice_x, layer]
            mask = exp > patch
            patch[mask] = exp[mask]

    def create_jitter_offset(self, joints, meta):
        """
        Generate ground-truth Jitter offset to the nearest keypoints on the shared two channels
        """
        assert meta['joint_num'] == joints.shape[1], 'num of joints mismatch'
        channel_num = meta['joint_num']
        offset_vectors = np.full((self.out_h, self.out_w, 2), np.inf, dtype=np.float32)

        for i in range(channel_num):
            visible = joints[:, i, 2] > 0  # only annotated (visible) keypoints are considered
            self.put_jitter_maps(offset_vectors, 0, joints[visible, i, 0:2])

        return offset_vectors.transpose((2, 0, 1))

    def put_jitter_maps(self, offset_vectors, layer, joints):  # 设置layer=0就可以保证offset放置在两个channel
        """
        Generate Jitter offset (delta_x, delta_y) on the 2*layer, 2*layer+1 channels
        """
        for i in range(joints.shape[0]):
            x_min = int(round(joints[i, 0] / self.stride - self.fill_jitter_size / 2))
            x_max = int(round(joints[i, 0] / self.stride + self.fill_jitter_size / 2))
            y_min = int(round(joints[i, 1] / self.stride - self.fill_jitter_size / 2))
            y_max = int(round(joints[i, 1] / self.stride + self.fill_jitter_size / 2))

            if y_max < 0:
                continue

            if x_max < 0:
                continue

            if x_min < 0:
                x_min = 0

            if y_min < 0:
                y_min = 0

            slice_x = slice(x_min, x_max)  # + 1
            slice_y = slice(y_min, y_max)

            # type: np.ndarray # joints[i, 0] -> x
            offset_x = joints[i, 0] - self.grid_x[slice_x].astype(np.float32)
            # type: np.ndarray # joints[i, 1] -> y
            offset_y = joints[i, 1] - self.grid_y[slice_y].astype(np.float32)
            offset_x_mesh = np.repeat(offset_x.reshape(1, -1), offset_y.shape[0], axis=0)
            offset_y_mesh = np.repeat(offset_y.reshape(-1, 1), offset_x.shape[0], axis=1)
            offset_mesh = np.concatenate(
                (offset_x_mesh[..., None], offset_y_mesh[..., None]),
                axis=-1)
            offset_mesh_l = np.linalg.norm(offset_mesh, axis=-1)

            offset_patch = offset_vectors[slice_y, slice_x, 2 * layer: 2 * layer + 2]
            vector_l = np.linalg.norm(offset_patch, axis=-1)

            mask = offset_mesh_l < vector_l

            # overlap the offset values on the basis of the offset lengths
            offset_patch[mask] = offset_mesh[mask]
