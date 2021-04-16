import logging
import os
import torch.utils.data
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

import transforms

LOG = logging.getLogger(__name__)


class CocoKeypoints(torch.utils.data.Dataset):
    """MSCOCO keypoint dataset. Custom our own dataset.

    Based on `torchvision.dataset.CocoDetection`.
    `MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        img_dir (string): root directory where images are downloaded to.
        annFile (string): path to json annotation file.
        preprocess (callable, optional): a function/transform that takes in an RGB numpy image
            (or annotation) and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transforms (callable, optional): a function/transform that takes in the
            image (and annotation) and generate target.
        n_images (int, optional): only using the first n_images if set.
        all_images (bool, optional): whether or not to use all the images.
        all_persons (bool, optional): only if ``all_images`` is False, then this works.
            Using all images containing persons.

    Attributes:
        cat_ids (list): filtered category ids
        ids (list): images filtered by cat_ids
    """

    def __init__(self, img_dir, annFile, *,
                 preprocess=None, target_transforms=None,
                 n_images=None,  all_images=False,
                 all_persons=False, shuffle=False,
                 debug_mask=False):
        from pycocotools.coco import COCO
        self.img_dir = img_dir
        self.coco = COCO(annFile)

        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        if all_images:
            self.ids = self.coco.getImgIds()
        elif all_persons:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
        else:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
            self.filter_for_keypoint_annotations()
        if n_images:
            self.ids = self.ids[:n_images]
        print(f'Path to annotation file: {annFile}')
        print('Evaluation Image Numbers: {}'.format(len(self.ids)))

        if shuffle:
            random.shuffle(self.ids)  # shuffle in place

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms
        self.debug_mask = debug_mask

    def filter_for_keypoint_annotations(self):
        """
        Returns:
            list: a list of image ids containing keypoint annotations. For example:
        """
        print('filter for keypoint annotations ...')

        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        print('... filter done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): index for image id

        Returns:
            tuple: tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)

        image_info = self.coco.loadImgs(image_id)[0]
        LOG.debug(image_info)
        image_path = os.path.join(self.img_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            raise IOError("image with current path dose not exist: %s" % image_path)

        image = cv2.imread(image_path)
        # We use RGB image sequence all through our project.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask_miss areas: 0, mask_all areas: 255
        mask_miss, _ = self.mask_mask(image_info, anns, debug_show=self.debug_mask)

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
            'image_path': image_path,
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta_init['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # transformation with augmentation on input image and annotations
        image, anns, meta, mask_miss = self.preprocess(image, anns, None, mask_miss)
        meta.update(meta_init)  # python dict update or add new entry

        # transform_targets generates ground truth
        if self.target_transforms is not None:
            anns = [t(anns, meta, mask_miss) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.ids)

    def mask_mask(self, image_info, anns, debug_show=False):
        """Mask all unannotated people (people crowd or persons having no keypoint annotation)
        as mask_miss, and mask all persons which are labeled as mask_all, in this whole image.

        Returns:
            tuple: mask_miss (OpenCV img) noted by 0, mask_all (OpenCV img) noted by 255.
            """
        h = image_info['height']
        w = image_info['width']
        mask_all = np.zeros((h, w), dtype=np.uint8)
        mask_miss = np.zeros((h, w), dtype=np.uint8)

        flag = 0

        # get all annotated masks
        for p in anns:
            if p["iscrowd"] == 1:  # handel crowd
                # segmentation annotation: crowd=0 'polygons'; iscrowd=1 'RLE'.
                mask_crowd = self.coco.annToMask(p)
                # temp is the IoU of the old mask_all and the current crowded instances mask
                temp = np.bitwise_and(mask_all, mask_crowd)
                mask_crowd = mask_crowd - temp

                flag += 1
                continue
            else:
                mask = self.coco.annToMask(p)

            # mask_all record all persons' masks in the current image
            mask_all = np.bitwise_or(mask, mask_all)
            # Note: small objects (segment area < 32^2) in COCO do not contain keypoint annotations,
            #   thus actually you needn't use "or p["area"] <= 32 * 32" (just for as reminder)
            if p["num_keypoints"] <= 0 or p["area"] <= 32 * 32:
                mask_miss = np.bitwise_or(mask, mask_miss)

        # make mask_miss and mask_all
        if flag < 1:
            mask_miss = np.logical_not(mask_miss)
        elif flag == 1:
            # mask the few keypoint and crowded persons at the same time ! mask areas are 0 !
            mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
            # mask all the persons including crowd, mask area are 1 !
            mask_all = np.bitwise_or(mask_all, mask_crowd)
        else:  # 对一个区域，只能存在一个segment,不存在一个区域同时属于某两个instances的部分
            raise Exception("crowd segments > 1")

        mask_miss = mask_miss.astype(np.uint8)
        mask_miss *= 255  # min=0, max=255

        mask_all = mask_all.astype(np.uint8)
        mask_all *= 255

        if debug_show:
            mask_concat = np.concatenate((mask_miss[:, :, np.newaxis], mask_all[:, :, np.newaxis]), axis=2)
            plt.imshow(np.repeat(mask_concat[:, :, 1][:, :, np.newaxis], 3, axis=2))  # mask_all
            plt.show()
            plt.imshow(np.repeat(mask_concat[:, :, 0][:, :, np.newaxis], 3, axis=2))  # mask_miss
            plt.show()
            LOG.debug('mask_miss_min %d, mask_miss_max %d',
                      mask_miss.min(), mask_miss.max())

        return mask_miss, mask_all


class ImageList(torch.utils.data.Dataset):
    """
     Args:
        image_paths (list): a list of image paths
        preprocess (Preprocess):  preprocessing the input image
    """
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = []  # for compatibility: there is no annotation
        image, anns, meta, _ = self.preprocess(image, anns, None, None)
        meta.update({
            'dataset_index': index,
            'file_name': image_path,
        })

        return image, anns, meta

    def __len__(self):
        return len(self.image_paths)
