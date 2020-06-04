import re
import logging
from .heatmap import HeatMaps
from .offset import OffsetMaps
from config.coco_data import (COCO_PERSON_SKELETON, COCO_PERSON_WITH_REDUNDANT_SKELETON,
                              REDUNDANT_CONNECTIONS, KINEMATIC_TREE_SKELETON)


LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('heatmap encoder')
    group.add_argument('--gaussian-clip-thre', default=HeatMaps.clip_thre,
                       type=float,
                       help='Gaussian distribution below this value is cut to zero')
    group.add_argument('--sigma', default=HeatMaps.sigma,
                       type=int, help='standard deviation of Gaussian distribution')

    group = parser.add_argument_group('offsetmap and scalemap encoder')
    group.add_argument('--fill-scale-size', default=OffsetMaps.fill_scale_size,
                       type=int,
                       help='the area around the keypoint will be filled with joint scale values.')
    group.add_argument('--min_scale', default=OffsetMaps.min_scale,
                       type=float, help='set minimum keypoint scale')


def factory(args, strides=None):
    """ Build ground truth encoders.
    """
    # configure heatmap
    if not strides:
        strides = [4, 4, 4]
    HeatMaps.clip_thre = args.gaussian_clip_thre
    HeatMaps.sigma = args.sigma

    # configure scalemap
    OffsetMaps.fill_scale_size = args.fill_scale_size
    OffsetMaps.min_scale = args.min_scale

    return factory_heads(args.headnets, args.square_length, strides)


def factory_heads(headnames, square_length, strides):
    """Build ground truth encoders.

    Args:
        headnames (list): a list of head names for encoders.
        square_length (int): square length of input images fed into the model.
        strides (list): a list of strides for multi-res outputs.
    """
    if isinstance(headnames[0], (list, tuple)):
        return [factory_heads(task_headnames, square_length, task_strides)
                for task_headnames, task_strides in zip(headnames, strides)]

    encoders = [factory_head(head_name, square_length, stride)
                for head_name, stride in zip(headnames, strides)]

    return encoders


def factory_head(head_name, square_length, stride):
    if head_name in ('hmp',
                     'hmps',
                     'heatmap',
                     'heatmaps') or \
       re.match('hmp[s]?([0-9]+)$', head_name) is not None:  # +: repeat one or more times

        m = re.match('hmp[s]?([0-9]+)$', head_name)  # $ the end of string match
        if m is not None:
            n_keypoints = int(m.group(1))
            LOG.debug('using %d keypoints', n_keypoints)
            LOG.info('not supported yet')

        else:
            n_keypoints = 17
            LOG.info('default COCO 17 keypoints are supported for now')

        LOG.info('selected encoder heatmap for %s with %d keypoints', head_name, n_keypoints)
        return HeatMaps(square_length, stride)

    if head_name in ('omp',
                     'omps'
                     'offset',
                     'offsets') or \
       re.match('omp[s]?([0-9]+)$', head_name) is not None:
        if head_name in ('omp', 'omp19', 'omps', 'offset', 'offsets'):
            n_keypoints = 17
            OffsetMaps.skeleton = COCO_PERSON_SKELETON  # 默认使用这个人体骨架
        elif head_name in ('omp16',):
            n_keypoints = 17
            OffsetMaps.skeleton = KINEMATIC_TREE_SKELETON
        elif head_name in ('omp25',):
            n_keypoints = 17
            OffsetMaps.skeleton = COCO_PERSON_WITH_REDUNDANT_SKELETON
        elif head_name in ('omp6', 'omps6'):
            n_keypoints = 17
            OffsetMaps.skeleton = REDUNDANT_CONNECTIONS
        else:
            raise Exception('unknown skeleton type of head')

        LOG.info('selected encoder Paf for %s', head_name)
        return OffsetMaps(square_length, stride)
        # 构造并返回Paf，用于生成ground truth paf
    raise Exception('unknown head to create an encoder: {}'.format(head_name))

