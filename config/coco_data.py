"""
Configuration for CrowdPose Dataset
Copied from openpifpaf
"""
import logging
from config import heatmap_hflip, offset_hflip, vector_hflip, HFLIP


LOG = logging.getLogger(__name__)

ANNOTATIONS_TRAIN = 'data/link2CrowdPose/json/crowdpose_train.json'
ANNOTATIONS_VAL = 'data/link2CrowdPose/json/crowdpose_val.json'
IMAGE_DIR_TRAIN = 'data/link2CrowdPose/images'
IMAGE_DIR_VAL = 'data/link2CrowdPose/images'

ANNOTATIONS_TESTDEV = 'data/link2CrowdPose/json/crowdpose_test.json'
ANNOTATIONS_TEST = 'data/link2CrowdPose/json/crowdpose_test.json'
IMAGE_DIR_TEST = 'data/link2CrowdPose/images'

COCO_KEYPOINTS = [
    'left_shoulder',  # 1
    'right_shoulder',  # 2
    'left_elbow',  # 3
    'right_elbow',  # 4
    'left_wrist',  # 5
    'right_wrist',  # 6
    'left_hip',  # 7
    'right_hip',  # 8
    'left_knee',  # 9
    'right_knee',  # 10
    'left_ankle',  # 11
    'right_ankle',  # 12
    'head',  # 13
    'neck',  # 14
]

LEFT_INDEX = [i for i, v in enumerate(COCO_KEYPOINTS) if v.startswith('l')]
RIGHT_INDEX = [i for i, v in enumerate(COCO_KEYPOINTS) if v.startswith('r')]

COCO_PERSON_SKELETON = [
    [12, 13],
    [13, 0],
    [13, 1],
    [0, 1],
    [6, 7],
    [0, 2],
    [2, 4],
    [1, 3],
    [3, 5],
    [0, 6],
    [1, 7],
    [6, 8],
    [8, 10],
    [7, 9],
    [9, 11]
]

COCO_PERSON_SKELETON_DOWNUP = [  # after simulation, we get the same results as COCO_PERSON_SKELETON
    (15, 13), ]  # Not implemented

COCO_PERSON_WITH_REDUNDANT_SKELETON = [
    (0, 1), ]  # Not implemented

DENSER_COCO_PERSON_SKELETON = [
    (0, 1), ]  # Not implemented

REDUNDANT_CONNECTIONS = [
    c
    for c in DENSER_COCO_PERSON_SKELETON
    if c not in COCO_PERSON_SKELETON
]

KINEMATIC_TREE_SKELETON = [
    (0, 1), (1, 3),  # left head
    (0, 2), (2, 4),
    (0, 5),
    (5, 7), (7, 9),  # left arm
    (0, 6),
    (6, 8), (8, 10),  # right arm
    (5, 11), (11, 13), (13, 15),  # left side
    (6, 12), (12, 14), (14, 16),
]

# sigmas:
# https://github.com/Jeff-sjtu/CrowdPose/blob/master/
# crowdpose-api/PythonAPI/crowdposetools/cocoeval.py#L223
COCO_PERSON_SIGMAS = [
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
    0.079,  # head
    0.079,  # neck
]


def draw_skeletons():
    import numpy as np
    from visualization import show
    coordinates = np.array([[
        [-1.4, 8.0, 2.0],  # 'left_shoulder',
        [1.4, 8.0, 2.0],  # 'right_shoulder',
        [-1.75, 6.0, 2.0],  # 'left_elbow',
        [1.75, 6.2, 2.0],  # 'right_elbow',
        [-1.75, 4.0, 2.0],  # 'left_wrist',
        [1.75, 4.2, 2.0],  # 'right_wrist',
        [-1.26, 4.0, 2.0],  # 'left_hip',
        [1.26, 4.0, 2.0],  # 'right_hip',
        [-1.4, 2.0, 2.0],  # 'left_knee',
        [1.4, 2.1, 2.0],  # 'right_knee',
        [-1.4, 0.0, 2.0],  # 'left_ankle',
        [1.4, 0.1, 2.0],  # 'right_ankle',
        [0.0, 10.3, 2.0],  # head
        [0.0, 9.3, 2.0],  # neck,
    ]])

    keypoint_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                            markersize=1, linewidth=6)

    with show.canvas('../docs/skeleton_crowdpose.png', figsize=(2, 5)) as ax:
        ax.set_axis_off()
        keypoint_painter.keypoints(ax, coordinates, skeleton=COCO_PERSON_SKELETON)


def print_associations():
    print('number of limb connections: ', len(COCO_PERSON_SKELETON))
    for j1, j2 in COCO_PERSON_SKELETON:
        print(COCO_KEYPOINTS[j1], '-', COCO_KEYPOINTS[j2])


if __name__ == '__main__':
    # for examination
    print(LEFT_INDEX)
    print(RIGHT_INDEX)
    print_associations()
    draw_skeletons()

    print(f'hflip indices of keypoints: {heatmap_hflip(COCO_KEYPOINTS, HFLIP)} \n')
    vector_hflip(COCO_KEYPOINTS, COCO_PERSON_SKELETON, HFLIP)
