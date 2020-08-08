import random

from .preprocess import Preprocess


class RandomApply(Preprocess):
    """
    Randomly apply the transform method (such as the Preprocess subclass).

    Args:
        transform (Preprocess): Transformation applied.
        probability (float): Probability to apply transforms.
    """
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, image, anns, meta, mask_miss):
        if random.uniform(0, 1) > self.probability:
            return image, anns, meta, mask_miss
        return self.transform(image, anns, meta, mask_miss)
