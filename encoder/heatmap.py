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
    clip_thre = 0.02  # Gaussian distribution below this value will be set to zero
    sigma = 3  # standard deviation of Gaussian distribution
    n_keypoints = 17
    keypoints = COCO_KEYPOINTS
    include_background = True  # background heatmap

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
                                self.sigma, self.clip_thre)

        heatmaps = hmps.create_heatmaps(anns, meta)

        mask_miss = cv2.resize(mask_miss, (0, 0),
                               fx=self.in_out_scale, fy=self.in_out_scale,
                               interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        # mask_miss area marked by 0.
        mask_miss = (mask_miss > 0.5)  # .astype(np.float32)  bool mask_miss

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
            hmp_reverse = 1 - np.amax(heatmaps, axis=2)

            return (
                torch.from_numpy(heatmaps.transpose((2, 0, 1))),
                torch.from_numpy(hmp_reverse[None, ...]),
                torch.from_numpy(mask_miss[None, ...])
            )
        else:
            return (
                torch.from_numpy(heatmaps.transpose((2, 0, 1))),
                torch.tensor([]),
                torch.from_numpy(mask_miss[None, ...])
            )


class HeatMapGenerator(object):
    """
    Generate the keypoint heatmaps and keypoint scale feature map.
    """

    def __init__(self, input_size, stride, sigma, clip_thre):

        self.in_w = input_size[0]
        self.in_h = input_size[1]
        self.stride = stride
        self.sigma = sigma
        self.double_sigma2 = 2 * self.sigma * self.sigma
        # set responses lower than gaussian threshold to 0.
        self.gaussian_clip_thre = clip_thre
        # get the gaussian peak spread
        self.gaussian_size = 2 * math.ceil(
            math.sqrt(-self.double_sigma2 * math.log(self.gaussian_clip_thre)))
        assert self.gaussian_clip_thre > 0 and self.gaussian_size > 0, 'should bigger than zero'
        # cached common parameters which same for all iterations and all pictures
        self.out_w = self.in_w // stride
        self.out_h = self.in_h // stride

        LOG.debug('Input image size: %d*%d, network output size: %d*%d',
                  self.in_w, self.in_h, self.out_w, self.out_h)

        # mapping coordinates into original input with cell center alignment.
        self.grid_x = np.arange(self.out_w)  # * stride + stride / 2 - 0.5  # x -> width
        self.grid_y = np.arange(self.out_h)  # * stride + stride / 2 - 0.5  # y -> height

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

        # add reverse keypoint gaussian heat map on the last background channel
        # heatmaps[:, :, -1] = 1 - np.amax(heatmaps[:, :, :-1], axis=2)

        return heatmaps

    def put_joints(self, heatmaps, joints, channel_num):

        for i in range(channel_num):
            # only annotated keypoints are considered !
            visible = joints[:, i, 2] > 0
            self.put_gaussian_peaks(heatmaps, i, joints[visible, i])

    def put_gaussian_peaks(self, heatmaps, layer, joints):
        """
        Generate ground truth on a single channel.
        """
        for i in range(joints.shape[0]):

            x_min = int(round(joints[i, 0] / self.stride) - self.gaussian_size // 2)
            x_max = int(round(joints[i, 0] / self.stride) + self.gaussian_size // 2)
            y_min = int(round(joints[i, 1] / self.stride) - self.gaussian_size // 2)
            y_max = int(round(joints[i, 1] / self.stride) + self.gaussian_size // 2)

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
            slice_x = slice(x_min, x_max + 1)
            slice_y = slice(y_min, y_max + 1)

            disk_x = self.grid_x[slice_x].astype(np.float32) - joints[i, 0] / self.stride
            disk_y = self.grid_y[slice_y].astype(np.float32) - joints[i, 1] / self.stride
            exp_x = np.exp(- disk_x ** 2 / np.array([self.double_sigma2]).astype(np.float32))
            exp_y = np.exp(- disk_y ** 2 / np.array([self.double_sigma2]).astype(np.float32))

            # np.outer of two vector with length M and N gets a matrix with shape M*N
            exp = np.outer(exp_y, exp_x)

            # # Laplace heatmap
            # dis = exp-(math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma)
            # dist = np.sqrt((self.X - joints[i, 0])**2 +
            #                (self.Y - joints[i, 1])**2) / 2.0 / self.sigma
            # np.where(dist > 4.6052, 1e8, dist)
            # exp = np.exp(-dist)

            # overlap Gaussian peaks by taking the maximum
            # Must use slice view to overlap original array!
            patch = heatmaps[slice_y, slice_x, layer]
            mask = exp > patch
            patch[mask] = exp[mask]
