import copy
import logging
import os
import torch.utils.data
import torchvision

from data import transforms
from data import CocoKeypoints
from data.transforms import utils


LOG = logging.getLogger(__name__)

ANNOTATIONS_TRAIN = 'data/link2COCO2017/annotations/person_keypoints_train2017.json'
ANNOTATIONS_VAL = 'data/link2COCO2017/annotations/person_keypoints_val2017.json'
IMAGE_DIR_TRAIN = 'data/link2COCO2017/train2017'
IMAGE_DIR_VAL = 'data/link2COCO2017/val2017'

ANNOTATIONS_TESTDEV = 'data/link2COCO2017/annotations_trainval_info/image_info_test-dev2017.json'
ANNOTATIONS_TEST = 'data/link2COCO2017/annotations_trainval_info/image_info_test2017.json'
IMAGE_DIR_TEST = 'data/link2COCO2017/test2017/'


class DataPrefetcher():
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


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--n-images-train', default=None, type=int,
                       help='number of images to sample from the train subset')
    group.add_argument('--n-images-val', default=None, type=int,
                       help='number of images to sample from the val subset')
    group.add_argument('--loader-workers', default=8, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=8, type=int,
                       help='batch size')
    group.add_argument('--force-shuffle', default=False, action='store_true',
                       help='force the dataset shuffle by hand')

    group = parser.add_argument_group('training parameters for warp affine')
    group.add_argument('--square-length', default=512, type=int,
                       help='square edge of input images')
    group.add_argument('--stride', default=4, type=int,
                       help='the ration of the input size to the output size')
    group.add_argument('--flip-prob', default=0.5, type=float,
                       help='the probability to flip the input image')
    group.add_argument('--max-rotate', default=40, type=float,)
    group.add_argument('--min-scale', default=0.7, type=float,
                       help='the lower bound of the relative'
                            ' image scale during augmentation')
    group.add_argument('--max-scale', default=1.3, type=float)
    group.add_argument('--max-translate', default=50, type=int,
                       help='the upper bound of shitting the image during augmentation')
    group.add_argument('--debug-affine-show', default=False, action='store_true',
                       help='show the transformed image and keyooints')


def train_factory(args, preprocess, target_transforms):
    if args.loader_workers is None:
        args.loader_workers = 0

    train_data = CocoKeypoints(
        img_dir=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images_train,
        shuffle=args.force_shuffle
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=not args.debug,
        pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        drop_last=True,)

    val_data = CocoKeypoints(
        img_dir=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images_val,
        shuffle=args.force_shuffle
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.loader_workers, drop_last=True,)

    return train_loader, val_loader


if __name__ == '__main__':  # for debug
    from time import time
    import timeit
    import argparse
    import encoder

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    train_cli(parser)
    encoder.cli(parser)
    args = parser.parse_args()
    args.headnets=['heatmaps', 'offsets']


    def test_augmentation_speed(train_client, show_image=True):
        start = time()
        batch = 0
        for index in range(train_client.__len__()):
            batch += 1

            image, anno, meta, mask_miss = [v for v in  # , offsets, mask_offset
                                 train_client.__getitem__(index)]
            t = 2
            #
            # # show the generated ground truth
            # if show_image:
            #     # show_labels = cv2.resize(meta.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_CUBIC)
            #     # # offsets = cv2.resize(offsets.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_NEAREST)
            #     # mask_miss = np.repeat(mask_miss.transpose((1, 2, 0)), 3, axis=2)
            #     # mask_miss = cv2.resize(mask_miss, image.shape[:2], interpolation=cv2.INTER_NEAREST)
            #     # image = cv2.resize(image, mask_miss.shape[:2], interpolation=cv2.INTER_NEAREST)
            #     plt.imshow(image[:, :, [2, 1, 0]])  # Opencv image format: BGR
            #     plt.imshow(meta.transpose((1, 2, 0))[:, :, 20], alpha=0.5)  # mask_all
            #     # plt.imshow(show_labels[:, :, 3], alpha=0.5)  # mask_all
            #     plt.show()
            #     t = 2
        print("produce %d samples per second: " % (batch / (time() - start)))  # about 70~80 FPS on MBP-13


    preprocess_transformations = [
        transforms.NormalizeAnnotations(),
        transforms.WarpAffineTransforms(args.square_length, aug_params=args, debug_show=False),
        transforms.RandomApply(transforms.AnnotationJitter(), 0.1),
    ]

    preprocess_transformations += [
        transforms.RandomApply(transforms.JpegCompression(), 0.1),
        transforms.RandomApply(transforms.ColorTint(), 0.2),
        transforms.ImageTransform(torchvision.transforms.ToTensor()),
        transforms.ImageTransform(
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])),

    ]
    preprocess = transforms.Compose(preprocess_transformations)

    target_transform = encoder.factory(args)

    val_client = CocoKeypoints(IMAGE_DIR_VAL, ANNOTATIONS_VAL,
                               preprocess=preprocess,
                               target_transforms=target_transform,
                               n_images=300, shuffle=True)

    # test the data generator
    print(timeit.timeit(stmt='test_augmentation_speed(val_client, False)',
                        setup="from __main__ import test_augmentation_speed;"
                              "from __main__ import val_client",
                        number=2))  # run fun number times
