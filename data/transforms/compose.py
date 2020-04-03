from .preprocess import Preprocess


class Compose(Preprocess):
    """Compose all the transformations into a single class.
    """
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list

    def __call__(self, image, anns, meta, mask_miss):
        for p in self.preprocess_list:
            image, anns, meta, mask_miss = p(image, anns, meta, mask_miss)

        return image, anns, meta, mask_miss
