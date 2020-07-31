import logging
import time
import torch
import torch.nn.functional as F


LOG = logging.getLogger(__name__)


def normalize_hmps():
    """todo: filter and smooth the heatmaps"""
    pass


def hmp_NMS(heat, kernel=3):  # , thre=0.001
    """
    NMS on keypoint Gaussian response heatmap (score map).
    Peak responses are reserved, compress non-peak responses to zero.

    Args:
        heat (Tensor): input tensor with shape  (N, H, W, C).
        kernel:
        thre: values below this threshold are filtered (abandoned, we move this into group.py).

    Returns:
        tensor (Tensor): a float tenor with shape (N, H, W, C),
            only peak values are preserved, most of the values are 0.
    """
    t0 = time.time()
    pad = (kernel - 1) // 2
    pad_heat = F.pad(heat, (pad, pad, pad, pad))  # , mode='reflect'
    hmax = F.max_pool2d(pad_heat, (kernel, kernel), stride=1, padding=0)
    keep_mask = (hmax == heat).float()  # * (heat >= thre).float()  # bool tensor -> float tensor
    LOG.debug('maxpool-NMS time in heatmap: %.6fs', time.time() - t0)
    return heat * keep_mask


def topK_channel(scores, K=40):  # y=0, x=1,2,3,4... may be preserved because the lack of high peaks
    """
    Collect top K peaks and corresponding coordinates on each heatmap channel.

    Notes:
        Top K may include very small even zero responses!
    """
    n, c, h, w = scores.shape
    topk_scores, topk_idxs = torch.topk(scores.view(n, c, -1), K)
    topk_ys = (topk_idxs / w)
    topk_xs = (topk_idxs % w)
    return topk_scores, topk_idxs, topk_ys, topk_xs


def joint_dets(hmps, k):
    """Select Top k candidate keypoints in heatmaps"""
    t0 = time.time()
    filtered_hmps = hmp_NMS(hmps)
    # shape of hm_score, hm_inds, topk_ys, topk_xs = [batch, 17, topk]
    dets = topK_channel(filtered_hmps, K=k)
    LOG.debug('TopK keypoint detection time: %.6fs', time.time() - t0)
    return dets
