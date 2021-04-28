import logging
import torch
from models import heads, networks, losses
from utils.util import boolean_string
import argparse

LOG = logging.getLogger(__name__)


def net_cli(parser):
    group = parser.add_argument_group('model configuration')
    group.add_argument('--initialize-whole', default=True, type=boolean_string,
                       help='randomly initialize the basenet and headnets, '
                            'just set it to True if you are not certain')
    group.add_argument('--checkpoint-whole', default=None, type=str,
                       help='the checkpoint path to the whole model (basenet+headnets)')

    group = parser.add_argument_group('base network configuration')
    group.add_argument('--basenet', default='hourglass104',
                       help='base network, e.g. hourglass4stage')
    group.add_argument('--two-scale', default=False, action='store_true',
                       help='to be implemented')
    group.add_argument('--multi-scale', default=False, action='store_true',
                       help='to be implemented')
    group.add_argument('--no-pretrain', dest='pretrained', default=True,
                       action='store_false',
                       help='create BaseNet without pretraining')
    group.add_argument('--basenet-checkpoint', default="weights/hourglass_104_renamed.pth",
                       type=str, help='Path to the pre-trained model and optimizer.')
    group = parser.add_argument_group('head network configuration')
    group.add_argument('--headnets', default=['hmp', 'omp'], nargs='+',
                       help='head networks')
    group.add_argument('--strides', default=[4, 4], nargs='+', type=int,
                       help='rations of the input to the output of basenet, '
                            'also the strides of all sub headnets. '
                            'Also, they determin the strides in encoder and decoder')
    group.add_argument('--max-stride', default=128, type=int, choices=[64, 128],
                       help='the max down-sampling stride through the network. ')
    group.add_argument('--include-spread', default=False, action='store_true',
                       help='add conv layers into the headnet to regress the spread_b '
                            'of Laplace distribution, you should set it to '
                            'True if you want to use laplace loss')
    group.add_argument('--include-background', default=False, action='store_true',
                       help='add conve layers to regress the heatmap of background channel')
    group.add_argument('--include-jitter-offset', default=False, action='store_true',
                       help='add conve layers to regress the jitter refinement offset to the nearest keypoint')
    group.add_argument('--include-scale', default=False, action='store_true',
                       help='add conve layers to regress the keypoint scales '
                            'in separate channels')

    group = parser.add_argument_group('loss configuration')
    group.add_argument('--lambdas', default=[1, 1, 100, 100, 0.01],
                       type=float, nargs='+',
                       help='learning task scaling factors for hmp_loss, bg_hmp_loss, jitter_off_loss, '
                            'offset_loss and scale_loss, directly multiplied, not averaged')
    group.add_argument('--stack-weights', default=[1, 1],
                       type=float, nargs='+',
                       help='loss weights for different stacks, weighted-sum averaged')
    group.add_argument('--hmp-loss', default='focal_l2_loss',
                       choices=['l2_loss', 'focal_l2_loss'],
                       help='loss for heatmap regression')
    group.add_argument('--jitter-offset-loss', default='offset_l1_loss',
                       choices=['offset_l1_loss', 'vector_l1_loss', 'offset_laplace_loss'],
                       help='loss for jitter offeset regression')
    group.add_argument('--offset-loss', default='offset_l1_loss',
                       choices=['offset_l1_loss', 'vector_l1_loss', 'offset_laplace_loss', 'offset_instance_l1_loss'],
                       help='loss for offeset regression')
    group.add_argument('--sqrt-re', default=False, action='store_true',
                       help='rescale the offset loss using torch.sqrt')
    group.add_argument('--scale-loss', default='scale_l1_loss',
                       choices=['scale_l1_loss'],
                       help='loss for keypoint scale regression')
    group.add_argument('--ftao', default=losses.TAU, type=float,
                       help='threshold between fore/background in focal L2 loss during training')
    group.add_argument('--fgamma', default=losses.GAMMA, type=float,
                       help='order of scaling factor in focal L2 loss during training')
    group.add_argument('--lmargin', default=losses.MARGIN, type=float,
                       help='offset length below this value will not be punished '
                            'during training when we rescale the offset loss by sqrt operation')


def model_factory(args):
    """Build the whole model from scratch from the args"""
    losses.TAU = args.ftao
    losses.GAMMA = args.fgamma
    losses.MARGIN = args.lmargin

    if 'hourglass' in args.basenet:
        # build the base network
        basenet, n_stacks, stride, max_stride, feature_dim = hourglass_from_scratch(
            args.basenet, args.pretrained, args.basenet_checkpoint, args.initialize_whole)

        # build the head networks
        assert stride == args.strides[0], 'strides mismatch'
        assert max_stride == args.max_stride, 'please reset the max_stride based on the network manually'
        headnets = heads.headnets_factory(
            args.headnets,
            n_stacks,
            args.strides,
            feature_dim,
            args.include_spread,
            args.include_background,
            args.include_jitter_offset,
            args.include_scale)
        # no pre-trained headnets, so we randomly initialize them
        if args.initialize_whole:
            headnets = [networks.initialize_weights(h) for h in headnets]

        lossfuncs = losses.lossfuncs_factory(
            args.headnets,
            n_stacks,
            args.stack_weights,
            args.hmp_loss,
            args.jitter_offset_loss,
            args.offset_loss,
            args.scale_loss,
            args.sqrt_re)

        model = networks.NetworkWrapper(basenet, headnets)

        return model, lossfuncs

    # if 'merge_hourglass' in args.basenet: implement other network structures

    raise Exception(f'unknown base network: {args.base_name}')


def hourglass_from_scratch(base_name, pretrained, basenet_checkpoint, initialize_whole):
    basenet, n_stacks, stride, max_stride, feature_channel = networks.basenet_factory(
        base_name)
    # initialize model params in-place, for not all params are old ones from pre-trained model
    if initialize_whole:
        basenet = networks.initialize_weights(basenet)

    if pretrained:
        basenet, _, _, _, _ = networks.load_model(
            basenet, basenet_checkpoint)

    LOG.info('select %s as the backbone, n_stacks=%d, stride=%d, max_stride=%d, feature_channels=%d',
             basenet.__class__.__name__, n_stacks, stride, max_stride, feature_channel)
    return basenet, n_stacks, stride, max_stride, feature_channel


def debug_parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--for-debug',
                        default=False,
                        action='store_true',
                        help='this parse is only for debug the code')

    net_cli(parser)
    args = parser.parse_args('--for-debug --lmargin 0.00002 --include-spread --include-background'.split())
    return args


if __name__ == '__main__':
    # for debug

    log_level = logging.DEBUG  # logging.INFO
    # set RootLogger
    logging.basicConfig(level=log_level)

    args = debug_parse_args()

    import sys

    LOG.info({
        'type': 'process',
        'argv': sys.argv,  # 返回一个list，包含运行程序本身的名字，以及用户输入的命令行中给予的参数
        'args': vars(args),  # args命名空间的值
    })
    t = sys.argv
    cmd = ' '.join(t)
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
