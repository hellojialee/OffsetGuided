import logging
import torch
from models import heads, networks, losses
import argparse

LOG = logging.getLogger(__name__)


def net_cli(parser):
    group = parser.add_argument_group('model configuration')
    group.add_argument('--checkpoint', default=None,
                       help='Path to the pre-trained model and optimizer.')

    group = parser.add_argument_group('base network configuration')
    group.add_argument('--basenet', default='hourglass104',
                       help='base network, e.g. hourglass4stage')
    group.add_argument('--two-scale', default=False, action='store_true',
                       help='to be implemented')
    group.add_argument('--multi-scale', default=False, action='store_true',
                       help='to be implemented')
    group.add_argument('--no-pretrain', dest='pretrained', default=True,
                       action='store_false',
                       help='create basenet without pretraining')

    group = parser.add_argument_group('head network configuration')
    group.add_argument('--headnets', default=['hmp', 'omp'], nargs='+',
                       help='head networks')
    group.add_argument('--strides', default=[4, 4], nargs='+', type=int,
                       help='rations of the input to the output of basenet, '
                            'also the strides of all sub headnets')
    group.add_argument('--include-spread', default=False, action='store_true',
                       help='add conv layers to regress the spread_b '
                            'of Laplace distribution, you should set it to '
                            'True if you chose laplace loss')
    group.add_argument('--include-background', default=False, action='store_true',
                       help='include the heatmap of background channel')
    group.add_argument('--include-scale', default=False, action='store_true',
                       help='add cone layers to regress the keypoint scales '
                            'in separate channels')

    group = parser.add_argument_group('loss configuration')
    group.add_argument('--lambdas', default=[1, 1, 1, 1],
                       type=float, nargs='+',
                       help='learning task wights')
    group.add_argument('--stack-weights', default=[1, 1],
                       type=float, nargs='+',
                       help='loss weights of different stacks')
    group.add_argument('--hmp-loss', default='focal_l2_loss',
                       choices=['l2_loss', 'focal_l2_loss'],
                       help='loss for heatmap regression')
    group.add_argument('--offset-loss', default='offset_laplace_loss',
                       choices=['offset_l1_loss', 'offset_laplace_loss'],
                       help='loss for offeset regression')
    group.add_argument('--scale-loss', default='scale_l1_loss',
                       choices=['scale_l1_loss'],
                       help='loss for keypoint scale regression')


def model_factory(args):
    """Build the whole model from scratch from the args"""

    if 'hourglass' in args.basenet:
        # build the base network
        basenet, n_stacks, stride, feature_dim = hourglass_from_scratch(
            args.basenet, args.pretrained)

        # build the head networks
        assert stride == args.strides[0], 'strides mismatch'
        headnets = heads.headnets_factory(
            args.headnets,
            n_stacks,
            args.strides,
            feature_dim,
            args.include_spread,
            args.include_background,
            args.include_scale)
        # no pre-trained headnets, so we randomly initialize them
        headnets = [networks.initialize_weights(h) for h in headnets]

        lossfuncs = losses.lossfuncs_factory(
            args.headnets, n_stacks, args.stack_weights,
            args.hmp_loss, args.offset_loss, args.scale_loss)
        model = networks.NetworkWrap(basenet, headnets)

        return model, lossfuncs

    # if 'merge_hourglass' in args.basenet: implement other network structures

    raise Exception(f'unknown base network: {args.base_name}')


def hourglass_from_scratch(base_name, pretrained):
    basenet, n_stacks, stride, feature_dim = networks.basenet_factory(
        base_name)
    # initialize model params in-place
    basenet = networks.initialize_weights(basenet)

    if pretrained:
        basenet, _, _ = networks.load_model(
            basenet, '../weights/hourglass_104_renamed.pth')

    return basenet, n_stacks, stride, feature_dim


def debug_parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--for-debug',
                        default=False,
                        action='store_true',
                        help='this parse is only for debug the code')

    net_cli(parser)
    args = parser.parse_args('--for-debug  --include-spread --include-background --include-scale'.split())
    return args


if __name__ == '__main__':
    # for debug

    log_level = logging.DEBUG  # logging.INFO
    # set RootLogger
    logging.basicConfig(level=log_level)

    args = debug_parse_args()
    model, lossfuns = model_factory(args)
    model.cuda()
    model.eval()  # 只有被正确注册的参数和子网络才会被自动移动到cuda上

    # for name, parameters in basenet.named_parameters():
    #     print(parameters)
    img = torch.rand(2, 3, 512, 512).cuda()
    out = model(img)

    print(len(out))

    hmp_loss, omp_loss = lossfuns
    gt_hmps = torch.rand(2, 17, 128, 128).cuda()
    gt_bghmp = torch.rand(2, 1, 128, 128).cuda()
    mask_miss = torch.ones((2, 1, 128, 128)).cuda()
    mask_miss = mask_miss > 0

    gt_offsets = torch.rand(2, 38, 128, 128).cuda()
    gt_scales = torch.rand(2, 17, 128, 128).cuda()

    loss1 = hmp_loss(out[0], gt_hmps, gt_bghmp, mask_miss)
    loss2 = omp_loss(out[1], gt_offsets, gt_scales, mask_miss)

    print(loss1, loss2)
    loss = sum(loss1) + sum(loss2)
    loss.backward()
    print('done')

