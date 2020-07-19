import torch
import torch.nn.functional as F


def normalize_hmps():
    """todo: filter and smooth the heatmaps"""
    pass


def hmp_NMS(heat, kernel=3, thre=0.001):
    """
    NMS on keypoint Gaussian response heatmap (score map).
    Peak responses are reserved, compress non-peak responses to zero.
    Args:
        heat (Tensor): input tensor with shape  (N, H, W, C).
        kernel:
        thre: values below this threshold are filtered.

    Returns:
        tensor (Tensor): a float tenor with shape (N, H, W, C),
            only peak values are preserved, most of the values are 0.
    """
    pad = (kernel - 1) // 2
    pad_heat = F.pad(heat, (pad, pad, pad, pad), mode='reflect')
    hmax = F.max_pool2d(pad_heat, (kernel, kernel), stride=1, padding=0)
    keep_mask = (hmax == heat).float() * (heat >= thre).float()  # bool tensor -> float tensor
    return heat * keep_mask


def topK_channel(scores, K=40):  # fixme: y=0, x=1,2,3,4等等边界点会保留下来，因为可能NMS之后没有这么多peak，所以有0峰值被保留了
    """
    Collect top K peaks and corresponding coordinates on each heatmap channel.
    Notes:

        Top K may include very small even zero responses!
    """
    n, c, h, w = scores.size()
    topk_scores, topk_idxs = torch.topk(scores.view(n, c, -1), K)
    topk_ys = (topk_idxs / w)
    topk_xs = (topk_idxs % w)
    return topk_scores, topk_idxs, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat