"""Dataloader and
Configurations for data preparations and transformations"""

import logging
import torch.utils.data
import torchvision

import transforms
from data import CocoKeypoints

LOG = logging.getLogger(__name__)

ANNOTATIONS_TRAIN = 'data/link2COCO2017/annotations/person_keypoints_train2017.json'
ANNOTATIONS_VAL = 'data/link2COCO2017/annotations/person_keypoints_val2017.json'
IMAGE_DIR_TRAIN = 'data/link2COCO2017/train2017'
IMAGE_DIR_VAL = 'data/link2COCO2017/val2017'

ANNOTATIONS_TESTDEV = 'data/link2COCO2017/annotations_trainval_info/image_info_test-dev2017.json'
ANNOTATIONS_TEST = 'data/link2COCO2017/annotations_trainval_info/image_info_test2017.json'
IMAGE_DIR_TEST = 'data/link2COCO2017/test2017/'


def collate_images_anns_meta(batch):
    # for more flexible label collection
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas


class DataPrefetcher(object):
    # Copied from: https://github.com/NVIDIA/apex/blob/b5a7c5f972/examples/imagenet/main_amp.py
    def __init__(self, loader, opt=None):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.opt = opt
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

        with torch.cuda.stream(self.stream):
            self.batch = [data_tensor.cuda(non_blocking=True)
                          for data_tensor in self.batch]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


def data_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--n-images-train', default=None, type=int,
                       help='number of images to sample from the trains subset')
    group.add_argument('--n-images-val', default=None, type=int,
                       help='number of images to sample from the val subset')
    group.add_argument('--loader-workers', default=8, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=8, type=int,
                       help='batch size')
    group.add_argument('--train-shuffle', default=False, action='store_true',
                       help='force the trains dataset shuffle by hand')
    group.add_argument('--val-shuffle', default=False, action='store_true',
                       help='force the validate dataset shuffle by hand')

    group = parser.add_argument_group('training parameters for warp affine')
    group.add_argument('--square-length', default=512, type=int,
                       help='square edge of input images')
    group.add_argument('--flip-prob', default=0.5, type=float,
                       help='the probability to flip the input image')
    group.add_argument('--max-rotate', default=45, type=float,  # 40
                       help='upper bound of the image rotation during augmentation')
    group.add_argument('--min-scale', default=0.5, type=float,
                       help='lower bound of the relative'
                            ' image scale during augmentation')
    group.add_argument('--max-scale', default=2.0, type=float,  # 1.5
                       help='upper bound of the relative'
                            ' image scale during augmentation')
    group.add_argument('--min-stretch', default=0.95, type=float,
                       help='lower bound of the relative'  # leads to less than 0.3% AP
                            ' image length stretch during augmentation')
    group.add_argument('--max-stretch', default=1.05, type=float,
                       help='upper bound of the relative'
                            ' image length stretch during augmentation')
    # bigger shift can produce cropped persons
    group.add_argument('--max-translate', default=150, type=int,  # 50
                       help='upper bound of shitting the image during augmentation')
    group.add_argument('--debug-affine-show', default=False, action='store_true',
                       help='show the transformed image and keyooints')


def dataloader_factory(args, preprocess, target_transforms=None):
    if target_transforms is None:
        collate_fn = collate_images_anns_meta
    else:
        collate_fn = collate_images_targets_meta

    if args.loader_workers is None:
        args.loader_workers = 0
    train_data, val_data = dataset_factory(args, preprocess, target_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=False,  # we control shuffle by args.train_shuffle
        pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        drop_last=True,
        collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        drop_last=True,
        collate_fn=collate_fn)

    return train_loader, val_loader


def dataset_factory(args, preprocess, target_transforms):
    train_data = CocoKeypoints(
        img_dir=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images_train,
        shuffle=args.train_shuffle  # shuffle the data
    )

    val_data = CocoKeypoints(
        img_dir=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images_val,
        shuffle=args.val_shuffle
    )

    return train_data, val_data


if __name__ == '__main__':  # for debug
    from time import time
    import timeit
    import argparse
    import encoder
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser(
        description=__doc__,
        # __doc__: current module's annotation (or module.a_function's annotation)
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # --help text with default values, e.g., gaussian threshold (default: 0.1)
        # RawTextHelpFormatter # --help text without default values, e.g., gaussian threshold
    )

    data_cli(parser)
    encoder.encoder_cli(parser)
    args = parser.parse_args()
    args.headnets = ['heatmaps', 'offsets']  # NOTICE : call net_cli before encoder_cli!!
    args.include_background = True  # generate the heatmap of background
    args.include_scale = True  # generate keypoint scale
    args.include_jitter_offset = True  # generate keypoint jitter offset

    log_level = logging.WARNING  # logging.INFO
    # set RootLogger
    logging.basicConfig(level=log_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    LOG.info('Test the data loader')


    def test_augmentation_speed(data_client, show_image=True):
        start = time()
        batch = 0
        for index in range(data_client.__len__()):
            batch += 1

            image, annos, meta = [v for v in
                                  data_client.__getitem__(index)]
            # mask_miss[0] from heatmap.py is actually the same as mask_miss[1] from offset.py
            image = image.numpy()
            mask_miss = annos[0][-1].numpy().astype(np.float32)  # bool -> float
            hmp = annos[0][0].numpy()
            jitter = annos[0][2].numpy()
            jitter[np.isinf(jitter)] = 0
            offset = annos[1][0].numpy()
            offset[np.isinf(offset)] = 0

            instance_scale = annos[1][2].numpy()
            instance_scale[np.isinf(instance_scale)] = 0

            # # show the generated ground truth
            if show_image:
                image = image.transpose((1, 2, 0))
                image = np.clip((image + 2.0) / 4.0, 0.0, 1.0)
                show_labels = cv2.resize(hmp.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_CUBIC)
                offset = cv2.resize(offset.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_LINEAR)
                jitter = cv2.resize(jitter.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_LINEAR)
                mask_miss = np.repeat(mask_miss.transpose((1, 2, 0)), 3, axis=2)
                mask_miss = cv2.resize(mask_miss, image.shape[:2], interpolation=cv2.INTER_NEAREST)
                plt.imshow(image)
                plt.imshow(mask_miss, alpha=0.5)  # mask_all
                plt.show()

                plt.imshow(image)  # We have manually set Opencv earlier: RGB
                plt.imshow(show_labels[:, :, 11], alpha=0.5)  # mask_all
                plt.show()

                plt.imshow(image)  # We have manually set Opencv earlier: RGB
                plt.imshow(jitter[:, :, 0], alpha=0.5)  # mask_all
                plt.show()

                plt.imshow(image)  # We have manually set Opencv earlier: RGB
                plt.imshow(jitter[:, :, 1], alpha=0.5)  # mask_all
                plt.show()
        print("produce %d samples per second: " % (batch / (time() - start)))  # about 70~80 FPS on MBP-13


    class AugParams:
        """
        An example of augmentation params.
        """
        flip_prob = 0.5
        max_rotate = 40
        min_scale = 0.7
        max_scale = 1.3
        max_translate = 50


    preprocess_transformations = [
        transforms.NormalizeAnnotations(),
        transforms.WarpAffineTransforms(args.square_length, aug_params=args, debug_show=False),
        # transforms.RandomApply(transforms.AnnotationJitter(), 0.1),
    ]

    preprocess_transformations += [
        # transforms.RandomApply(transforms.JpegCompression(), 0.1),
        transforms.RandomApply(transforms.ColorTint(), 0.2),
        transforms.ImageTransform(torchvision.transforms.ToTensor()),
        transforms.ImageTransform(
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])),

    ]
    preprocess = transforms.Compose(preprocess_transformations)

    args.strides = [4, 4]

    args.pin_memory = False
    if torch.cuda.is_available():
        args.pin_memory = True

    target_transform = encoder.encoder_factory(args, args.strides)

    val_client = CocoKeypoints(IMAGE_DIR_VAL, ANNOTATIONS_VAL,
                               preprocess=preprocess,
                               target_transforms=target_transform,
                               n_images=1000, shuffle=True)

    # test the data generator
    print(timeit.timeit(stmt='test_augmentation_speed(val_client, True)',  # True
                        setup="from __main__ import test_augmentation_speed;"
                              "from __main__ import val_client",
                        number=3))  # run for number times  # generate 17 samples per second
