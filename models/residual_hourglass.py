"""
Still in development.
"""
import math
import torch
from torch import nn
from .layers import Conv, Hourglass, SELayer, Backbone


MultiTaskLossParallel = nn.Sequential()  # do nothing to the input tensor
MultiTaskLoss = nn.Sequential()


class IMHNOpt:
    nstack = 4  # stacked number of hourglass
    hourglass_inp_dim = 256
    increase = 128  # increased channels once down-sampling in the hourglass networks
    nstack_weight = [1, 1, 1, 1]  # weight the losses between different stacks, stack 1, stack 2, stack 3...
    scale_weight = [0.1, 0.2, 0.4, 1.6,
                    6.4]  # weight the losses between different scales, scale 128, scale 64, scale 32...
    multi_task_weight = 0.1  # person mask loss vs keypoint loss
    keypoint_task_weight = 3  # keypoint heatmap loss vs body part heatmap loss
    # Download the pre-trained model snapshotted at epoch 52 first.
    ckpt_path = './link2checkpoints_distributed/PoseNet_52_epoch.pth'


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim, bn=False):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=bn)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        # Regress 5 different scales of heatmaps per stack
        self.before_regress = nn.ModuleList(
            [nn.Sequential(Conv(inp_dim + i * increase, inp_dim, 3, bn=bn, dropout=False),
                           Conv(inp_dim, inp_dim, 3, bn=bn, dropout=False),
                           # ##################### Channel Attention layer  #####################
                           SELayer(inp_dim),
                           ) for i in range(5)])

    def forward(self, fms):
        assert len(fms) == 5, "hourglass output {} tensors,but 5 scale heatmaps are supervised".format(len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):
    """
    Pack or initialize the trainable parameters of the network

    Attributes:
        nstack: number of stack
        inp_dim: input tensor channels fed into the hourglass block
        oup_dim: channels of regressed feature maps
        bn: use batch normalization
        increase: increased channels once down-sampling
        init_weights:
        **kwargs:
    """
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, init_weights=True, **kwargs):
        super(PoseNet, self).__init__()
        # self.pre = nn.Sequential(
        #     Conv(3, 64, 7, 2, bn=bn),
        #     Conv(64, 128, bn=bn),
        #     nn.MaxPool2d(2, 2),
        #     Conv(128, 128, bn=bn),
        #     Conv(128, inp_dim, bn=bn)
        # )
        self.pre = Backbone(nFeat=inp_dim)  # It doesn't affect the results regardless of which self.pre is used
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase, bn=bn) for _ in range(nstack)])
        # predict 5 different scales of heatmpas per stack, keep in mind to pack the list using ModuleList.
        # Notice: nn.ModuleList can only identify Module subclass! Thus, we must pack the inner layers in ModuleList.
        # TODO: change the outs layers, Conv(inp_dim + j * increase, oup_dim, 1, relu=False, bn=False)
        self.outs = nn.ModuleList(
            [nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for j in range(5)]) for i in
             range(nstack)])

        # TODO: change the merge layers, Merge(inp_dim + j * increase, inp_dim + j * increase)
        self.merge_features = nn.ModuleList(
            [nn.ModuleList([Merge(inp_dim, inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
             range(nstack - 1)])
        self.merge_preds = nn.ModuleList(
            [nn.ModuleList([Merge(oup_dim, inp_dim + j * increase, bn=bn) for j in range(5)]) for i in range(nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        # Input Tensor: a batch of images within [0,1], shape=(N, H, W, C). Pre-processing was done in data generator
        x = imgs.permute(0, 3, 1, 2)  # Permute the dimensions of images to (N, C, H, W)
        x = self.pre(x)
        pred = []
        # loop over stack
        for i in range(self.nstack):
            preds_instack = []
            # return 5 scales of feature maps
            hourglass_feature = self.hourglass[i](x)

            if i == 0:  # cache for smaller feature maps produced by hourglass block
                features_cache = [torch.zeros_like(hourglass_feature[scale]) for scale in range(5)]

            else:  # residual connection across stacks
                #  python里面的+=, ，*=也是in-place operation,需要注意
                hourglass_feature = [hourglass_feature[scale] + features_cache[scale] for scale in range(5)]
            # feature maps before heatmap regression
            features_instack = self.features[i](hourglass_feature)

            for j in range(5):  # handle 5 scales of heatmaps
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])  # input tensor for next stack
                        features_cache[j] = self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])

                    else:
                        # reset the res caches
                        features_cache[j] = self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
            pred.append(preds_instack)
        # returned list shape: [nstack * [batch*128*128, batch*64*64, batch*32*32, batch*16*16, batch*8*8]]z
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            # 卷积的初始化方法
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # Kaiming 初始化在这里会梯度爆炸, 方差为2/n. math.sqrt(2. / n)
                m.weight.data.normal_(0, 0.001)    # # math.sqrt(2. / n)
                # torch.nn.init.kaiming_normal_(m.weight) or 直接使用现成的nn.init中的函数。
                # bias都初始化为0
                if m.bias is not None:  # 当有BN层时，卷积层Con不加bias！
                    m.bias.data.zero_()
            # batchnorm使用全1初始化 bias全0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """
    def __init__(self, opt, out_dim, bn=False, dist=False, swa=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, out_dim, bn=bn, increase=opt.increase)
        # If we use train_parallel, we implement the parallel loss . And if we use train_distributed,
        # we should use single process loss because each process on these 4 GPUs  is independent
        self.criterion = MultiTaskLoss if dist else MultiTaskLossParallel
        self.swa = swa

    def forward(self, input_all):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        inp_imgs = input_all[0]
        target_tuple = input_all[1:]
        output_tuple = self.posenet(inp_imgs)

        if not self.training:  # testing mode
            loss = self.criterion(output_tuple, target_tuple)
            # output will be concatenated  along batch channel automatically after the parallel model return
            return output_tuple, loss

        else:  # training mode
            if not self.swa:
                loss = self.criterion(output_tuple, target_tuple)

                # output will be concatenated  along batch channel automatically after the parallel model return
                return loss
            else:
                return output_tuple


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """
    def __init__(self, opt, out_dim, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, out_dim, bn=bn, init_weights=False,
                               increase=opt.increase)

    def forward(self, inp_imgs):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        output_tuple = self.posenet(inp_imgs)

        if not self.training:
            # output will be concatenated  along batch channel automatically after the parallel model return
            return output_tuple
        else:
            # output will be concatenated  along batch channel automatically after the parallel model return
            raise ValueError('\nOnly eval mode is available!!')


if __name__ == '__main__':
    from time import time
    net_opt = IMHNOpt()
    out_dim = 55
    model = NetworkEval(net_opt, out_dim, bn=True)

    img = torch.rand(1, 3, 512, 512)

    print('Resuming from checkpoint ...... ')
    checkpoint = torch.load(net_opt.ckpt_path, map_location=torch.device('cpu'))  # map to cpu to save the gpu memory
    model.load_state_dict(checkpoint['weights'])  # 加入他人训练的模型，可能需要忽略部分层，则strict=False
    print('Network weights have been resumed from checkpoint...')

    if torch.cuda.is_available():
        model.cuda()

    model.eval()  # set eval mode is important

    # import torch.onnx
    #
    # torch.onnx.export(model, img, "hourglass-104.onnx")
    # # ############################# netron --host=localhost --port=8080
    #
    # # # # ##############  Count the FLOPs of your PyTorch model  ##########################
    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(img,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    # ---------------------------------------- #
    # Hourglass_104 234.269G, 187.700M
    # Ours 4-stage IMHN 269.882G, 128.999M
    # Ours 3-stage IMHN 206.546G, 96.676M
    # ---------------------------------------- #
