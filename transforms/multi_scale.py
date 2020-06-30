from .preprocess import Preprocess


class MultiScale(Preprocess):
    def __init__(self, preprocess_list):
        """Create lists of preprocesses.

        Used in generating multi-scale labels.

        Must be the most outer preprocess function.
        Preprocess_list can contain transforms.Compose() functions.
        """
        self.preprocess_list = preprocess_list

    def __call__(self, image, anns, meta, mask_miss):
        image_list, anns_list, meta_list, mask_miss_list = [], [], [], []
        for p in self.preprocess_list:
            this_image, this_anns, this_meta, this_mask_miss = p(image, anns, meta, mask_miss)
            image_list.append(this_image)
            anns_list.append(this_anns)
            meta_list.append(this_meta)
            mask_miss_list.append(this_mask_miss)

        return image_list, anns_list, meta_list, mask_miss_list
