import logging
import torch
import re

LOG = logging.getLogger(__name__)

TAU = 0.01  # threshold between fore/background in focal L2 loss during training
GAMMA = 1  # order of scaling factor in focal L2 loss during training
MARGIN = 0.1  # offset length below this value will not be punished


def l1(x, t):
    """Compute the L1 loss of two tensors."""
    return torch.abs(x - t).sum()


def l2(x, t):
    out = torch.mul((x - t) ** 2, 0.5)
    return out.sum()


def laplace(norm, logb):
    out = 0.693147 + logb + norm * torch.exp(-logb)
    return out.sum()


def focal_l2(s, sxing, tau=TAU, gamma=GAMMA):
    st = torch.where(torch.ge(sxing, tau), s, 1. - s)
    factor = torch.abs(1. - st) ** gamma
    out = torch.mul((s - sxing) ** 2 * factor, 0.5)
    return out.sum()


def tensor_loss(pred, gt, mask_miss, fun):
    """ Compute the distance of two tensors with mask_miss and finite_mask.
    Args:
        pred: tensor shape (N, C, out_h, out_w)
        gt: tensor shape (N, C, out_h, out_w)
        mask_miss: tensor shape (N, 1, out_h, out_w)"""
    # # Notice! expand does not allocate more memory but just m
    # ake the tensor look as if you expanded it. You should call
    # .clone() or repeat on the resulting tensor if you plan on modifying it
    # https://discuss.pytorch.org/t/very-strange-behavior-change-one-element-of-a-tensor-will-influence-all-elements/41190
    mask_miss = mask_miss.expand_as(gt)
    labelled_pred = pred[mask_miss]
    labelled_gt = gt[mask_miss]

    # return true if tensor is not infinity or not Not a Number (nan etc,)
    mask = torch.isfinite(labelled_gt)

    return fun(labelled_pred[mask], labelled_gt[mask])


class LossChoice(object):
    @staticmethod
    def l2_loss(pred, gt, mask_miss):
        return tensor_loss(pred, gt, mask_miss, l2)

    @staticmethod
    def focal_l2_loss(pred, gt, mask_miss):
        return tensor_loss(pred, gt, mask_miss, focal_l2)

    @staticmethod
    def scale_l1_loss(pred, gt, mask_miss):
        """
        Args:
            pred: inferred tensor of shape (N, C, out_h, out_w)
            gt: groudtruth tensor of shape (N, C, out_h, out_w)
            mask_miss: tensor of shape (N, 1, out_h, out_w) unlabelled areas denoted as 0,
            fun: function to compute the loss.
        """
        return tensor_loss(pred, gt, mask_miss, l1)

    @staticmethod
    def offset_l1_loss(pred, gt, _, mask_miss):
        return tensor_loss(pred, gt, mask_miss, l1)

    @staticmethod
    def offset_laplace_loss(pred, gt, logb, mask_miss):
        """
        Args:
            pred: inferred offset tensor, shape=(N, C, out_h, out_w)
            gt: ground truth offset tesor, shape=(N, C, out_h, out_w)
            logb: inferred laplace spread, shape=(N, C/2, out_h, out_w)
            mask_miss: shape=(1, out_h, out_w)
        """
        # spread logb: shape=(N, C//2, out_h, out_w)

        delta = (pred - gt)
        offset_x = delta[:, ::2, None, :, :]  # (N, C//2, 1， out_h, out_w)
        offset_y = delta[:, 1::2, None, :, :]  # (N, C//2, 1， out_h, out_w)

        norm = torch.cat(
            (offset_x, offset_y), dim=2).norm(
            dim=2)  # (N, C//2, out_h, out_w)

        labelled_norm = norm[mask_miss.expand_as(norm)]
        labelled_logb = logb[mask_miss.expand_as(logb)]
        labelled_offset_x = offset_x.squeeze(
        )[mask_miss.expand_as(offset_x.squeeze())]

        mask = torch.isfinite(labelled_offset_x)
        return laplace(labelled_norm[mask], labelled_logb[mask])


class HeatMapsLoss(object):

    def __init__(self, head_name, n_stacks, stack_weights, hmp_loss):
        super(HeatMapsLoss, self).__init__()

        self.head_name = head_name + '_loss'
        self.n_stacks = n_stacks
        assert len(stack_weights) >= n_stacks, type(stack_weights)
        self.stack_weights = [weight / sum(stack_weights) for weight in stack_weights]
        self.hmp_loss = hmp_loss

        LOG.debug('%s loss config: n_stacks = %d, stack_weights = %s, loss = %s ',
                  head_name, n_stacks, stack_weights, self.hmp_loss.__name__)

    def __call__(self, pred_hpms, gt_hpm, gt_bghmp, mask_miss):
        """
        Args:
            pred_hpms: inferred outputs like [hmp_stack1, hmp_stack2], [bg_hmp_stack1, bg_hmp_stack2]
            gt_hpm:
            mask_miss:
        Returns:
             keypoints hmp loss, background hmp loss
        """

        assert len(pred_hpms[0]) == self.n_stacks, 'BaseNet mismatches HeadNet'
        batch_size = gt_hpm.shape[0]
        LOG.debug('batch size = %d', batch_size)
        hmps, bg_hmps = pred_hpms
        out1, out2 = [], []

        for stack_i, (hmp, bg_hmp) in enumerate(zip(hmps, bg_hmps)):  # loop each stack

            weighted_loss = torch.mul(self.hmp_loss(hmp, gt_hpm, mask_miss), self.stack_weights[stack_i])
            out1.append(weighted_loss)

            if len(bg_hmp) > 0:  # background heatmap loss
                weighted_bgloss = torch.mul(self.hmp_loss(bg_hmp, gt_bghmp, mask_miss), self.stack_weights[stack_i])
                out2.append(weighted_bgloss)
        LOG.debug('hmp loss: %s, \t background hmp loss: %s', out1, out2)
        return sum(out1) / batch_size, sum(out2) / batch_size


class OffsetMapsLoss(object):

    def __init__(self, head_name, n_stacks, stack_weights, off_loss, s_loss, sqrt_re=False):
        super(OffsetMapsLoss, self).__init__()
        assert len(stack_weights) >= n_stacks, type(stack_weights)

        self.head_name = head_name + '_loss'
        self.n_stacks = n_stacks
        self.stack_weights = [weight / sum(stack_weights) for weight in stack_weights]
        self.off_loss = off_loss
        self.s_loss = s_loss
        self.sqrt_re = sqrt_re  # resize the offset loss by log

        LOG.debug('%s loss config: n_stacks = %d, stack_weights = %s, losses = %s, %s, ',
                  head_name, n_stacks, stack_weights,
                  self.off_loss.__name__, self.s_loss.__name__)

    def __call__(self, preds, gt_off, gt_s, mask_miss):
        """
        Returns: offset loss (of vectors of x, y), keypoint scale loss
        """
        assert len(preds[0]) == self.n_stacks
        batch_size = gt_off.shape[0]
        LOG.debug('batch size = %d', batch_size)
        out1, out2 = [], []
        pred_off_stacks, pred_spread_stacks, pred_scale_stacks = preds

        for stack_i, (pred_off, pred_spread, pred_s) in enumerate(
                zip(pred_off_stacks, pred_spread_stacks, pred_scale_stacks)):
            inter1 = torch.mul(
                self.off_loss(pred_off, gt_off, pred_spread, mask_miss),
                self.stack_weights[stack_i])
            if self.sqrt_re:
                inter1 = torch.sqrt(inter1 + MARGIN)  # fixme
            out1.append(inter1)

            if len(pred_s) > 0:
                inter2 = torch.mul(
                    self.s_loss(pred_s, gt_s, mask_miss),
                    self.stack_weights[stack_i])
                out2.append(inter2)
        LOG.debug('connection offset loss: %s, \t keypoint scale loss: %s', out1, out2)
        return sum(out1) / batch_size, sum(out2) / batch_size


def lossfuncs_factory(headnames, n_stacks, stack_weights,
                      heatmap_loss, offset_loss, scale_loss, sqrt_re):
    """Build loss networks.

    Args:
        headnames (list): a list of head names, e.g., "hmp17 omp19"
        n_stacks (int): base network may has multiple output tensors from all stacks.
        stack_weights (list): weights for different stacks.
    """
    hmp_loss = getattr(LossChoice, heatmap_loss)

    off_loss = getattr(LossChoice, offset_loss)
    s_loss = getattr(LossChoice, scale_loss)

    lossnets = [factory_loss(h, n_stacks, stack_weights, hmp_loss, off_loss, s_loss, sqrt_re)
                for h in headnames]

    return lossnets


def factory_loss(head_name, n_stacks, stack_weights, hmp_loss, off_loss, s_loss, sqrt_re):
    """
    Build a head network.

    Args:
    """
    if head_name in ('hmp',
                     'hmps',
                     'heatmap',
                     'heatmaps') or \
            re.match('hmp[s]?([0-9]+)$', head_name) is not None:
        LOG.info('select loss net for %s head', head_name)
        return HeatMapsLoss(head_name, n_stacks, stack_weights, hmp_loss)

    if head_name in ('omp',
                     'omps'
                     'offset',
                     'offsets') or \
            re.match('omp[s]?([0-9]+)$', head_name) is not None:
        LOG.info('select loss net for %s head', head_name)
        return OffsetMapsLoss(head_name, n_stacks, stack_weights, off_loss, s_loss, sqrt_re)
        # 构造并返回Paf，用于生成ground truth paf
    raise Exception('unknown head to create a lossnet: {}'.format(head_name))


if __name__ == '__main__':
    gt_hmps = torch.rand(2, 17, 128, 128).cuda()
    gt_bghmp = torch.rand(2, 1, 128, 128).cuda()
    mask_miss = torch.ones((2, 1, 128, 128)).cuda()
    mask_miss = mask_miss > 0

    gt_offsets = torch.rand(2, 38, 128, 128).cuda()
    gt_scales = torch.rand(2, 17, 128, 128).cuda()

    loss = LossChoice.l2_loss
    loss2 = getattr(LossChoice, 'l2_loss')
    out = loss(gt_hmps, gt_hmps, mask_miss)
    out2 = loss2(gt_hmps, gt_hmps, mask_miss)
    print(out)
