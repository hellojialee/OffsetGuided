import numpy as np


def soft_nms(subset, suppressed_v=0):
    if not len(subset):
        return subset

    occupied = np.zeros((
        len(subset[0]),
        int(max(np.max(ann[:, 1]) for ann in subset) + 1),
        int(max(np.max(ann[:, 0]) for ann in subset) + 1),
    ), dtype=np.uint8)

    for ann in subset:
        joint_scales = np.maximum(10.0, ann[:, 3])

        assert len(occupied) == len(ann)
        for xyv, occ, joint_s in zip(ann[:, :3], occupied, joint_scales):
            v = xyv[2]
            if v == -1:
                continue
            # Use cython to speed up if needed：
            # https://python3-cookbook.readthedocs.io/zh_CN/latest/c15/p11_use_cython_to_write_high_performance_array_operation.html
            x = np.clip(xyv[0], 0.0, occ.shape[1] - 1).astype(int)
            y = np.clip(xyv[1], 0.0, occ.shape[0] - 1).astype(int)
            if occ[y, x]:
                xyv[2] = suppressed_v
            else:
                scalar_square_add_single(occ, xyv[0], xyv[1], joint_s, 1)
    return subset


def scalar_square_add_single(field, x, y, width, value):
    minx = max(0, int(x - width))
    miny = max(0, int(y - width))
    maxx = max(minx + 1, min(field.shape[1], int(x + width) + 1))
    maxy = max(miny + 1, min(field.shape[0], int(y + width) + 1))
    field[miny:maxy, minx:maxx] += value