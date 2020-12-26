"""Head networks. Regress heatmaps, offset and scale, etc."""

import logging
import torch
import re

LOG = logging.getLogger(__name__)


class HeatMapsHead(torch.nn.Module):
    stride = 4  # the output stride of the base network
    n_keypoints = 17
    include_background = False  # background heatmap
    bg_channel = 1
    include_jitter_offset = False  # jitter offsetmaps
    jo_channel = 2
    include_spread = False

    def __init__(self, head_name, inp_dim, n_stacks,
                 kernel_size=1, padding=0, dilation=1):
        super(HeatMapsHead, self).__init__()

        LOG.debug('%s config: inp_dim = %d, n_stacks = %d, stride= %d, '
                  'include background heatmap: %s, include jitter offsetmap: %s '
                  'n_keypoints = %d, kernel = %d, padding = %d, dilation = %d',
                  head_name, inp_dim, n_stacks, self.stride,
                  self.include_background, self.include_jitter_offset,
                  self.n_keypoints, kernel_size, padding, dilation)

        self.head_name = head_name
        self.n_stacks = n_stacks
        self.hp_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(inp_dim, self.n_keypoints,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_stacks)
        ])
        self.bghp_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(inp_dim, self.bg_channel,
                            kernel_size, padding=padding, dilation=dilation)
            if self.include_background else torch.nn.Sequential()
            for _ in range(n_stacks)])
        self.jitter_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(inp_dim, self.jo_channel,
                            kernel_size, padding=padding, dilation=dilation)
            if self.include_jitter_offset else torch.nn.Sequential()
            for _ in range(n_stacks)])

    def forward(self, args):
        assert len(args) == self.n_stacks, 'multiple outputs from BaseNet'
        out_hmps = []
        out_bghmps = []
        out_jomps = []

        for hmp_layer, bg_layer, jo_layer, x in zip(self.hp_convs, self.bghp_convs, self.jitter_convs, args):
            hmp = hmp_layer(x)
            out_hmps.append(hmp)

            if self.include_background:
                bg_hmp = bg_layer(x)
                out_bghmps.append(bg_hmp)
            else:
                out_bghmps.append([])

            if self.include_jitter_offset:
                jitter_off = jo_layer(x)
                out_jomps.append(jitter_off)
            else:
                out_jomps.append([])

        return out_hmps, out_bghmps, out_jomps


class OffsetMapsHead(torch.nn.Module):
    stride = 4  # the output stride of the base network
    n_keypoints = HeatMapsHead.n_keypoints
    n_skeleton = 19
    include_spread = False  # learning with uncertainty
    include_scale = False

    def __init__(self, head_name, inp_dim, n_stacks,
                 kernel_size=1, padding=0, dilation=1):
        super(OffsetMapsHead, self).__init__()

        LOG.debug('%s config: inp_dim = %d, n_stacks = %d, stride = %d, '
                  'include spread regression: %s, include scale regression: %s, '
                  'n_skeleton = %d, kernel = %d, padding = %d, dilation = %d',
                  head_name, inp_dim, n_stacks, self.stride,
                  self.include_spread, self.include_scale,
                  self.n_skeleton, kernel_size, padding, dilation)

        self.head_name = head_name
        self.n_stacks = n_stacks

        # regression for offset vector x, y
        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(inp_dim, 2 * self.n_skeleton,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_stacks)
        ])

        # spread_b
        self.spread_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(inp_dim, self.n_skeleton,
                            kernel_size, padding=padding, dilation=dilation)
            if self.include_spread else torch.nn.Sequential()
            for _ in range(n_stacks)])

        # regression for keypoint scale
        self.scale_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(inp_dim, self.n_keypoints,
                            kernel_size, padding=padding, dilation=dilation)
            if self.include_scale else torch.nn.Sequential()
            for _ in range(n_stacks)
        ])

    def forward(self, args):
        assert len(args) == self.n_stacks, 'multiple outputs from BaseNet'
        out_offsets = []
        out_scales = []
        out_spreads = []

        for reg_layer, scale_layer, spread_layer, x in zip(
                self.reg_convs, self.scale_convs, self.spread_convs, args):

            offset = reg_layer(x)
            out_offsets.append(offset)

            if self.include_spread:
                spread = spread_layer(x)  # fixme: PIFAF use Leaky ReLu to do what?

                # spread = torch.nn.functional.relu(spread + 2) - 1.99  # spread是lnb, 大于0
                out_spreads.append(spread)
            else:
                out_spreads.append([])

            if self.include_scale:
                scale = scale_layer(x)
                out_scales.append(scale)
            else:
                out_scales.append([])

        return out_offsets, out_spreads, out_scales


def headnets_factory(headnames, n_stacks, strides, inp_dim,
                     include_spread, include_background, include_jitter, include_scale):
    """Build head networks.

    Args:
        headnames (list): a list of head names, e.g., "hmp17 omp19"
        n_stacks (int): base network may has multiple output tensors from all stacks.
        strides: the output strides of the base network.
        inp_dim: the tensor channels output by the base network.
        include_jitter: include the jitter offset to the nearest keypoints
        include_spread: used in laplace loss
        include_background: add the heatmap of background
    """

    headnets = [factory_head(h, n_stacks, s, inp_dim, include_spread, include_background, include_jitter, include_scale)
                for h, s in zip(headnames, strides)]

    return headnets


def factory_head(head_name, n_stacks, stride, inp_dim,
                 include_spread=False, include_background=False,
                 include_jitter=False, include_scale=False):
    """
    Build a head network.

    Args:
        include_spread (bool): regress the spread b of Laplace distribution
    """
    if head_name in ('hmp',
                     'hmps',
                     'heatmap',
                     'heatmaps') or \
            re.match('hmp[s]?([0-9]+)$', head_name) is not None:

        m = re.match('hmp[s]?([0-9]+)$', head_name)
        if m is not None:
            n_keypoints = int(m.group(1))
            LOG.info('using %d keypoints to generate heatmaps', n_keypoints)
            assert n_keypoints == 17, f'{n_keypoints} keypoint not supported'

        else:
            n_keypoints = 17

        LOG.info('select HeatMapsHead of stride %d to infer %d keypoint', stride, n_keypoints)
        HeatMapsHead.stride = stride
        HeatMapsHead.n_keypoints = n_keypoints
        HeatMapsHead.include_background = include_background
        HeatMapsHead.include_jitter_offset = include_jitter
        return HeatMapsHead(head_name, inp_dim, n_stacks)

    if head_name in ('omp',
                     'omps'
                     'offset',
                     'offsets') or \
            re.match('omp[s]?([0-9]+)$', head_name) is not None:

        m = re.match('omp[s]?([0-9]+)$', head_name)
        if m is not None:
            n_skeleton = int(m.group(1))
            LOG.info(
                'using %d skeleton connections to generate offsetmaps',
                n_skeleton)
            assert n_skeleton in [
                19, 16, 31, 44, 25], 'check skeleton configuration'

        else:
            n_skeleton = 19

        LOG.info('select OffsetMapsHead of stride %d to infer %d skeleton connections',
                 stride, n_skeleton)
        OffsetMapsHead.stride = stride
        OffsetMapsHead.n_skeleton = n_skeleton
        OffsetMapsHead.include_spread = include_spread
        OffsetMapsHead.include_scale = include_scale
        return OffsetMapsHead(head_name, inp_dim, n_stacks)
        # 构造并返回Paf，用于生成ground truth paf
    raise Exception('unknown head to create an encoder: {}'.format(head_name))


if __name__ == '__main__':
    headnames = ['hmp', 'omp']

    head_nets = headnets_factory(headnames, 2, [4, 4], 256, True, True, True, True)
    t = id(head_nets[0].n_keypoints) == id(head_nets[1].n_keypoints)
    head_nets[1].n_keypoints = 20
    t2 = id(head_nets[0].n_keypoints) == id(head_nets[1].n_keypoints)
    print(head_nets)
