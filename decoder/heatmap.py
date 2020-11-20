import logging
import time
import math
import numpy as np
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


def topK_channel(filtered_scores, hmps, K=40):  # y=0, x=1,2,3,4... may be preserved because the lack of high peaks
    """
    Collect top K peaks and corresponding coordinates on each heatmap channel.

    Notes:
        Top K may include very small even zero responses!
    """
    n, c, h, w = filtered_scores.shape
    topk_scores, topk_idxs = torch.topk(filtered_scores.view(n, c, -1), K)
    topk_ys = (topk_idxs / w)
    topk_xs = (topk_idxs % w)

    #  ##########  The least squares estimate for keypoint ###########
    # topk_ys, topk_xs = get_final_preds(hmps, topk_xs, topk_ys, kernel=1)
    # ##########################################

    return topk_scores, topk_idxs, topk_ys, topk_xs


def joint_dets(hmps, k):
    """Select Top k candidate keypoints in heatmaps"""
    t0 = time.time()
    filtered_hmps = hmp_NMS(hmps)
    # shape of hm_score, hm_inds, topk_ys, topk_xs = [batch, 17, topk]
    dets = topK_channel(filtered_hmps, hmps, K=k)
    LOG.debug('TopK keypoint detection time: %.6fs', time.time() - t0)

    return dets


def get_point(hm,coords,k=1):
    A = []
    B = []
    xm = int(coords[0])
    ym = int(coords[1])
    tmp = hm[ym][xm]
    for i in range(2*k+1):
        for j in range(2*k+1):
            py = coords[1] - k
            px = coords[0] - k
            px = (px+i).astype(int)
            py = (py+j).astype(int)
            if(min(hm.shape[1] - px-1 , px , hm.shape[0] - py-1 , py)<0 ):
                    continue
            hm[py][px] /= tmp
            hm[py][px] = max(min(hm[py][px],1),1e-8)
            A.append(np.array([2*xm-2*px, 2*ym-2*py, 2*math.log(hm[py][px]) ]))
            B.append(np.array([xm**2+ym**2-px**2-py**2]))
    if(len(A)<2):
        return coords
    else:
        A = np.array(A)
        B = np.array(B)
        X = np.dot(A.T,A)
        X = np.linalg.pinv(X)
        X =  np.dot(np.dot(X,A.T),B)
        coords = np.array([X[0][0],X[1][0]])
        return coords


def get_final_preds(batch_heatmaps, topk_xs0, topk_ys0, kernel=1):
    batch_heatmaps = batch_heatmaps.cpu().numpy()
    topk_xs = topk_xs0.cpu().numpy()[..., np.newaxis]
    topk_ys = topk_ys0.cpu().numpy()[..., np.newaxis]

    coords = np.concatenate((topk_xs, topk_ys), axis=-1)

    # heatmap_height = batch_heatmaps.shape[2]
    # heatmap_width = batch_heatmaps.shape[3]

    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            for j in range(coords.shape[2]):  # j-th keypoint in the current heatmap
                hm = batch_heatmaps[n][p]
                # px = int(math.floor(coords[n][p][0]))
                # py = int(math.floor(coords[n][p][1]))
                px = int(round(coords[n][p][j][0]))
                py = int(round(coords[n][p][j][1]))
                # k = min(heatmap_width - px-1 , px , heatmap_height-py-1  , py,2)
                coords[n][p][j] = get_point(hm, np.array([px, py]), kernel)
                # k = 2
                # coords[n][p] = get_point1(hm,np.array([px,py]),k)
                # if k < px < heatmap_width-k and k < py < heatmap_height-k:   #对求出的最大值的点作微分
                #     coords[n][p] = get_point(hm,np.array([px,py]),k)
                # elif 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                #     coords[n][p] = get_point1(hm,np.array([px,py]))

    return torch.tensor(coords[..., 1], device=topk_ys0.device), \
           torch.tensor(coords[..., 0], device=topk_xs0.device)
