"""Transform input data.
 Import functions from this package can facilitate reference in other package.

OpenCV-resize() regards the pixels lies at the center of cells in the image array.
The origin (0, 0) of the coordinate system lies at the center of the op-left cell.

OpenCV-warpaffine() regards the pixels as dots without areas, thus in this case the origin
(0, 0) of the coordinate system lies at the top-left pixel.

A rectangle is denoted as the format (x,y,w,h), e.g. (0, 0, 800, 600).
"""

import torchvision

from .annotations import AnnotationJitter, NormalizeAnnotations
from .compose import Compose
from .image import Blur, ImageTransform, JpegCompression, ColorTint, Gray
from .pad import CenterPad, RightDownPad, SquarePad
from .random import RandomApply
from .scale import RescaleLongAbsolute, RescaleHighAbsolute, RescaleRelative
from .affine import WarpAffineTransforms
from .affine import FixedAugParams


EVAL_TRANSFORM = Compose([
    NormalizeAnnotations(),
    # ToTensor: Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    #  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    ImageTransform(torchvision.transforms.ToTensor()),
    ImageTransform(
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ),
])


# 注意：有些变换操作只支持的PIL图像对象，而有些操作只支持Pytorch Tensor，两者混用时注意先完成类型转换
TRAIN_TRANSFORM = Compose([
    NormalizeAnnotations(),
    ImageTransform(torchvision.transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)),
    RandomApply(JpegCompression(), 0.1),  # maybe irrelevant for COCO, but good for others RandomApply(Blur(), 0.01),
    # maybe irrelevant for COCO, but good for others. https://github.com/vita-epfl/openpifpaf/issues/113
    # It's to add jpeg artifacts as a data augmentation. It seems to help when applying the models
    #  to other data sets that are heavily jpeg compressed.
    # ImageTransform(torchvision.transforms.RandomGrayscale(p=0.01)),
    EVAL_TRANSFORM,
])
