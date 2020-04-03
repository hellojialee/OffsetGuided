"""
Configurations for keypoint, skeleton and keypoint jitter sigmas.
"""


COCO_PERSON_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

COCO_PERSON_WITH_REDUNDANT_SKELETON = []

REDUNDANT_CONNECTIONS = [
    c
    for c in COCO_PERSON_WITH_REDUNDANT_SKELETON
    if c not in COCO_PERSON_SKELETON
]

KINEMATIC_TREE_SKELETON = [   # fixme: index form 0
    (0, 1), (1, 3),  # left head
    (0, 2), (2, 4),
    (0, 5),
    (5, 7), (7, 9),  # left arm
    (0, 6),
    (6, 8), (8, 10),  # right arm
    (5, 11), (11, 13), (13, 15),  # left side
    (6, 12), (12, 14), (14, 16),
]


COCO_KEYPOINTS = [
    'nose',            # 0
    'left_eye',        # 1
    'right_eye',       # 2
    'left_ear',        # 3
    'right_ear',       # 4
    'left_shoulder',   # 5
    'right_shoulder',  # 6
    'left_elbow',      # 7
    'right_elbow',     # 8
    'left_wrist',      # 9
    'right_wrist',     # 10
    'left_hip',        # 11
    'right_hip',       # 12
    'left_knee',       # 13
    'right_knee',      # 14
    'left_ankle',      # 15
    'right_ankle',     # 16
]

LEFT_INDEX = [i for i, v in enumerate(COCO_KEYPOINTS) if v.startswith('l')]
RIGHT_INDEX = [i for i, v in enumerate(COCO_KEYPOINTS) if v.startswith('r')]

COCO_PERSON_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
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
]

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


DENSER_COCO_PERSON_SKELETON = [   # fixme: index form 0
    (1, 2), (1, 3), (2, 3), (1, 4), (1, 5), (4, 5),
    (1, 6), (1, 7), (2, 6), (3, 7),
    (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
    (6, 12), (7, 13), (6, 13), (7, 12), (12, 13),
    (6, 8), (7, 9), (8, 10), (9, 11), (6, 10), (7, 11),
    (8, 9), (10, 11),
    (10, 12), (11, 13),
    (10, 14), (11, 15),
    (14, 12), (15, 13), (12, 15), (13, 14),
    (12, 16), (13, 17),
    (16, 14), (17, 15), (14, 17), (15, 16),
    (14, 15), (16, 17),
]


DENSER_COCO_PERSON_CONNECTIONS = [
    c
    for c in DENSER_COCO_PERSON_SKELETON
    if c not in COCO_PERSON_SKELETON
]


def print_associations():
    for j1, j2 in COCO_PERSON_SKELETON:
        print(COCO_KEYPOINTS[j1], '-', COCO_KEYPOINTS[j2])


if __name__ == '__main__':
    print_associations()
