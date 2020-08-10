import io
import logging

import numpy as np
from PIL import Image
import scipy
from scipy import ndimage
import torch
import cv2
import torchvision

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class ImageTransform(Preprocess):
    """
    To be compatible with transforms from other packages,
    such as torchvision.transforms.ColorJitter,
    so that we can introduce them into our transforms.
    """
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, image, anns, meta, mask_miss):
        image = self.image_transform(image)
        return image, anns, meta, mask_miss


class JpegCompression(Preprocess):
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, image, anns, meta, mask_miss):
        image = Image.fromarray(image)
        f = io.BytesIO()
        image.save(f, 'jpeg', quality=self.quality)
        PIL_img = Image.open(f)
        image = np.array(PIL_img)
        return image, anns, meta, mask_miss


class Blur(Preprocess):
    def __init__(self, max_sigma=5.0):
        self.max_sigma = max_sigma

    def __call__(self, image, anns, meta, mask_miss):
        im_np = np.asarray(image)
        sigma = self.max_sigma * float(torch.rand(1).item())
        im_np = scipy.ndimage.filters.gaussian_filter(im_np, sigma=(sigma, sigma, 0))
        return im_np, anns, meta, mask_miss


class Gray(Preprocess):
    def __init__(self):
        self.transform = torchvision.transforms.RandomGrayscale(p=1)

    def __call__(self, image, anns, meta, mask_miss):
        image = Image.fromarray(image)
        image = self.transform(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 单通道？
        image = np.asarray(image)

        return image, anns, meta, mask_miss


class ColorTint(Preprocess):

    def __call__(self, image, anns, meta, mask_miss):
        # uint8 input，OpenCV outputs Hue、Saturation、Value ranges are: [0,180)，[0,256)，[0,256)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)

        # hue
        hsv_img[:, :, 0] = np.maximum(
            np.minimum(hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), 179), 0)
        # saturation
        hsv_img[:, :, 1] = np.maximum(
            np.minimum(hsv_img[:, :, 1] - 40 + np.random.randint(80 + 1), 255), 0)
        # value
        hsv_img[:, :, 2] = np.maximum(
            np.minimum(hsv_img[:, :, 2] - 30 + np.random.randint(60 + 1), 255), 0)

        hsv_img = hsv_img.astype(np.uint8)
        image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return image, anns, meta, mask_miss
