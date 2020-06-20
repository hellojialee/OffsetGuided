import logging
import torch
from models import heads, networks
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

    group = parser.add_argument_group('loss configuration')
    group.add_argument('--lambdas', default=[1, 1, 1],
                       type=float, nargs='+',
                       help='learning task wights')
    group.add_argument('--stack-weights', default=[1, 1],
                       type=float, nargs='+',
                       help='loss weights of different stacks')
    group.add_argument('--hmp-loss', default='l2',
                       choices=['l2', 'focall2'],
                       help='loss for heatmap regression')
    group.add_argument('--tau', default=0.01, type=float,
                       help='background threshold in focal L2 loss')
    group.add_argument('--offset-loss', default='l1',
                       choices=['smoothl1', 'l1', 'laplace'],
                       help='loss for offeset regression')
    group.add_argument('--scale-loss', default='l1',
                       choices=['smoothl1', 'l1', 'laplace'],
                       help='loss for keypoint scale regression')


def net_factory(args):
    # Build the whole model from scratch
    if 'hourglass' in args.basenet:
        basenet, headnets = hourglass_from_scratch(
            args.basenet, args.headnets, args.strides, args.pretrained)
    # elif: implement other network structures

    else:
        raise Exception(f'unknown base network: {args.base_name}')

    lossnet = torch.nn.Sequential

    # modelpkg = networks.Network(basenet, headnets, lossnet)  # todo: to complete the loss

    return basenet, headnets # modelpkg


def hourglass_from_scratch(base_name, head_names, head_strides, pretrained):
    basenet, n_stacks, stride, feature_dim = networks.build_basenet(base_name)
    assert stride == head_strides[0], 'strides mismatch'
    # initialize model params in-place
    basenet = networks.initialize_weights(basenet)

    if pretrained:
        basenet, _, _ = networks.load_model(
            basenet, '../weights/hourglass_104_renamed.pth')

    headnets = heads.headnets_factory(
        head_names, n_stacks, stride, feature_dim)
    # no pre-trained headnets, so we randomly initialize them
    headnets = [networks.initialize_weights(h) for h in headnets]

    return basenet, headnets


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--for-debug',
                        default=False,
                        action='store_true',
                        help='this parse is only for debug the code')

    net_cli(parser)
    args = parser.parse_args('--for-debug '.split())

    return args


if __name__ == '__main__':
    # for debug

    args = parse_args()
    basenet, headnets = net_factory(args)
    model = networks.NetworkEval(basenet, headnets)
    model.cuda()
    model.eval()  # 只有被正确注册的参数和子网络才会被自动移动到cuda上

    # for name, parameters in basenet.named_parameters():
    #     print(parameters)

    img = torch.rand(1, 3, 512, 512).cuda()
    out = model(img)
    print(len(out))
