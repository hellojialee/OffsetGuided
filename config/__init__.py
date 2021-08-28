"""Basic data configures"""
import logging

LOG = logging.getLogger(__name__)


coco_mean = [0.40789654, 0.44719302, 0.47026115]
coco_std = [0.28863828, 0.27408164, 0.27809835]

data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]

HFLIP = {
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}


def heatmap_hflip(keypoints, hflip=None):
    if hflip is None:
        hflip = HFLIP
    flip_indices = list([
        keypoints.index(hflip[kp_name]) if kp_name in hflip else kp_i
        for kp_i, kp_name in enumerate(keypoints)
    ])
    LOG.debug('hflip indices: %s', flip_indices)
    return flip_indices


def offset_hflip(keypoints, skeleton, hflip=None):
    if hflip is None:
        hflip = HFLIP
    skeleton_names = [
        (keypoints[j1], keypoints[j2])
        for j1, j2 in skeleton
    ]

    flipped_skeleton_names = [
        (hflip[j1] if j1 in hflip else j1, hflip[j2] if j2 in hflip else j2)
        for j1, j2 in skeleton_names
    ]
    LOG.debug(f'skeleton = {skeleton_names} \n flipped_skeleton = {flipped_skeleton_names}')

    flip_indices = list(range(len(skeleton)))
    reserve_indices = []
    for limb_i, (n1, n2) in enumerate(skeleton_names):
        if (n1, n2) in flipped_skeleton_names:
            flip_indices[limb_i] = flipped_skeleton_names.index((n1, n2))
        if (n2, n1) in flipped_skeleton_names:
            flip_indices[limb_i] = flipped_skeleton_names.index((n2, n1))
            reserve_indices.append(limb_i)
    LOG.debug(f'limb hflip indices: {flip_indices} \n limb reverse indices: {reserve_indices}')
    return flip_indices, reserve_indices


def vector_hflip(keypoints, skeleton, hflip=None):
    if hflip is None:
        hflip = HFLIP
    skeleton_names = [
        (keypoints[j1], keypoints[j2])
        for j1, j2 in skeleton
    ]
    flipped_skeleton_names = [
        (hflip[j1] if j1 in hflip else j1, hflip[j2] if j2 in hflip else j2)
        for j1, j2 in skeleton_names
    ]
    print(f'skeleton = {skeleton_names} \n flipped_skeleton = {flipped_skeleton_names}')

    flip_indices = list(range(len(skeleton)))
    reverse_direction = []
    for limb_i, (n1, n2) in enumerate(skeleton_names):
        if (n1, n2) in flipped_skeleton_names:
            flip_indices[limb_i] = flipped_skeleton_names.index((n1, n2))
        if (n2, n1) in flipped_skeleton_names:
            flip_indices[limb_i] = flipped_skeleton_names.index((n2, n1))
            reverse_direction.append(limb_i)
    print(f'hflip indices: {flip_indices} \n reverse: {reverse_direction}')
    return flip_indices, reverse_direction
