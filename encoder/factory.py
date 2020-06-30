import re
import logging
from .heatmap import HeatMaps
from .offset import OffsetMaps
from config.coco_data import (COCO_PERSON_SKELETON, COCO_PERSON_WITH_REDUNDANT_SKELETON,
                              REDUNDANT_CONNECTIONS, KINEMATIC_TREE_SKELETON)

LOG = logging.getLogger(__name__)


def encoder_cli(parser):
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


def encoder_factory(args, strides=None):
    """ Build ground truth encoders.
    """
    # properties outside __init__(): https://blog.csdn.net/xiaojiajia007/article/details/104434696
    # configure heatmap
    if not strides:
        strides = [4, 4, 4]
    HeatMaps.clip_thre = args.gaussian_clip_thre
    HeatMaps.sigma = args.sigma
    HeatMaps.include_background = args.include_background  # defined in head.py

    # configure scalemap
    OffsetMaps.fill_scale_size = args.fill_scale_size
    OffsetMaps.min_scale = args.min_scale
    OffsetMaps.include_scale = args.include_scale  # defined in head.py

    return factory_heads(args.headnets, args.square_length, strides)


def factory_heads(headnames, square_length, strides):
    """Build ground truth encoders.

    Args:
        headnames (list): a list of head names for encoders.
        square_length (int): square length of input images fed into the model.
        strides (list): a list of strides for multi-res outputs of the model.
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
        # example [ho]: the first letter must be 'h' or 'o
        # [s]?: a single s exists or not,
        # [s]+: one or more s exist

        m = re.match('hmp[s]?([0-9]+)$', head_name)  # $ the end of string match
        if m is not None:
            n_keypoints = int(m.group(1))  # group eg: hmps17, hmps, 17
            LOG.info('using %d keypoints to generate heatmaps', n_keypoints)
            assert n_keypoints == 17, f'{n_keypoints} keypoints not supported'

        else:
            n_keypoints = 17

        LOG.info('selected encoder: Heatmap for %s with %d keypoints', head_name, n_keypoints)
        HeatMaps.n_keypoints = n_keypoints
        return HeatMaps(square_length, stride)

    if head_name in ('omp',
                     'omps'
                     'offset',
                     'offsets') or \
            re.match('omp[s]?([0-9]+)$', head_name) is not None:
        if head_name in ('omp', 'omp19', 'omps', 'offset', 'offsets'):
            n_keypoints = 17
            OffsetMaps.skeleton = COCO_PERSON_SKELETON  # default selection
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

        LOG.info('selected encoder: Offset for %s', head_name)
        return OffsetMaps(square_length, stride)
        # 构造并返回Paf，用于生成ground truth paf
    raise Exception('unknown head to create an encoder: {}'.format(head_name))
