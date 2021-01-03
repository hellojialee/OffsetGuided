import numpy as np
import math
import numbers
import torch
from torch import nn


def scored_offset(hmp, off, jtypes_f, jtypes_t, kernel_size=7):
    """
    Refine offsets with the heatmap responses at start joint area.
    Args:
        hmp (Tensor): with shape (n, c, h, w)
        off (Tensor): with shape (n, l, h, w)
        jtypes_f (list): offset start joints
        jtypes_t (list): offset end joints
        kernel_size: weighted area

    Usage:
         # Construct the start_joints list and then feed the original hmps and offs Tensors
         joints_f, joints_t = decoder.offset.pack_jtypes(skeleton)
         offs = decoder.scored_offset(hmps, offs, joints_f, joints_t)

    Returns:
        weighted_offs (Tensor)
    """
    score_map = hmp[:, jtypes_f, :, :].unsqueeze(2)  # (n, limbs, h, w)  -->  (n, limbs, 1, h, w)
    n, limbs, h, w = off.size()
    offset_reshape = off.view((n, -1, 2, h, w))  # (n, limbs, 2, h, w)
    somap = score_map * offset_reshape  # (n, limbs, 2, h, w)

    mean_score = nn.functional.avg_pool2d(score_map.squeeze(),  # (n, limbs, h, w)
                                          kernel_size,
                                          stride=1,
                                          padding=(kernel_size - 1) // 2,
                                          divisor_override=1)  # divisor_override changes the denominator
    somap_temp = nn.functional.avg_pool2d(somap.view((n, -1, h, w)),  # (n, limbs*2, h, w)
                                          kernel_size,
                                          stride=1,
                                          padding=(kernel_size - 1) // 2,
                                          divisor_override=1)
    weighted_offset = somap_temp.view((n, -1, 2, h, w)) / (mean_score.unsqueeze(2) + 1e-6)

    return weighted_offset.view((n, -1, h, w))  # # (n, limbs, 2, h, w) --> (n, limbs*2, h, w)


def pack_jtypes(skeleton):
    jtypes_f, jtypes_t = [], []
    for i, (j_f, j_t) in enumerate(skeleton):
        jtypes_f.append(j_f)
        jtypes_t.append(j_t)
    return jtypes_f, jtypes_t