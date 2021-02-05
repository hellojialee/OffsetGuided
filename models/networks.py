import logging
import warnings
import os
import sys
import torch
import torch.nn as nn
from models import Hourglass104, Hourglass4Stage

LOG = logging.getLogger(__name__)


def load_model(model, ckpt_path, *, optimizer=None, drop_layers=True, drop_name='offset_convs',
               resume_optimizer=True, optimizer2cuda=True, load_amp=False):
    """
    Load pre-trained model and optimizer checkpoint.

    Args:
        model:
        ckpt_path: 
        optimizer: 
        drop_layers (bool): drop pre-trained params of the output layers, etc
        drop_name: drop layers with this string in names
        resume_optimizer:
        optimizer2cuda (bool): move optimizer statues to cuda
        load_amp (bool): load the amp state including loss_scalers
            and their corresponding unskipped steps
    """

    start_epoch = 0
    start_loss = float('inf')
    if not os.path.isfile(ckpt_path):
        print(f'WARNING!! ##### Current checkpoint file {ckpt_path} DOSE NOT exist!!#####')
        warnings.warn("No pre-trained parameters are loaded!"
                      " Please make sure you initialize the model randomly!")
        user_choice = input("Are you sure want to continue with randomly model (y/n):\n")
        if user_choice in ('y', 'Y'):
            # return without loading
            load_amp = False
            return model, optimizer, start_epoch, start_loss, load_amp
        else:
            sys.exit(0)

    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    LOG.info('Loading pre-trained model %s, checkpoint at epoch %d', ckpt_path,
             checkpoint['epoch'])
    start_epoch = checkpoint['epoch'] + 1
    start_loss = checkpoint['train_loss']
    state_dict_ = checkpoint['model_state_dict']  # type: dict
    if load_amp and 'amp' in checkpoint.keys():
        LOG.info('Found saved amp state including loss_scalers and their corresponding '
                 'unskipped steps from checkpoint %s at epoch %d', ckpt_path, start_epoch)
        load_amp = checkpoint['amp']
    else:
        print(f'No OLD amp state is detected from current checkpoint {ckpt_path} '
              f'or you do not load amp state')
        load_amp = False

    from collections import OrderedDict
    state_dict = OrderedDict()  # loaded pre-trained model weight

    # convert parallel/distributed model to single model
    for k, v in state_dict_.items():  # Fixme: keep consistent with our model
        if (drop_name in k or 'some_example_convs' in k) and drop_layers:  #
            continue
        if k.startswith('module') and not k.startswith('module_list'):
            name = k[7:]  # remove prefix 'module.'
            # name = 'module.' + k  # add prefix 'module.'
            state_dict[name] = v
        else:
            name = k
            state_dict[name] = v
    model_state_dict = model.state_dict()  # newly built model

    # check loaded parameters and created model parameters
    msg1 = 'If you see this, your model does not fully load the ' + \
           'pre-trained weight. Please make sure ' + \
           'you have correctly built the model layers or the weight shapes.'
    msg2 = 'If you see this, your model has more parameters than the ' + \
           'pre-trained weight. Please make sure ' + \
           'you have correctly specified more layers.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                LOG.debug(
                    'Skip loading pre-trained parameter %s, current model '
                    'required shape %s, loaded shape %s. %s',
                    k, model_state_dict[k].shape, state_dict[k].shape, msg1)
                state_dict[k] = model_state_dict[k]  # fix badly mismatched params
        else:
            LOG.debug('Drop pre-trained parameter %s which current model dose '
                      'not have. %s', k, msg1)
    for k in model_state_dict:
        if not (k in state_dict):
            LOG.debug('No param %s in pre-trained model. %s', k, msg2)
            state_dict[k] = model_state_dict[k]  # append missing params to rescue
    model.load_state_dict(state_dict, strict=False)
    print(f'Network {model.__class__.__name__} weights have been resumed from checkpoint: {ckpt_path}')

    # resume optimizer parameters
    if optimizer is not None and resume_optimizer:
        if 'optimizer_state_dict' in checkpoint:
            LOG.debug('Resume the optimizer.')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Here, we must convert the resumed state data of optimizer to gpu.
            # In this project, we use map_location to map the state tensors to cpu.
            # In the training process, we need cuda version of state tensors,
            # so we have to convert them to gpu.
            if torch.cuda.is_available() and optimizer2cuda:
                LOG.debug('Move the optimizer states into GPU.')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            # param_group['lr'] will be instead set in a separate fun: adjust_learning_rate()
            print('Optimizer {} has been resumed from the checkpoint at epoch {}.'
                  .format(optimizer.__class__.__name__, start_epoch - 1))
        elif optimizer is not None:
            print('Optimizer {} is NOT resumed, although the checkpoint exists.'.format(optimizer.__class__.__name__))
        else:
            print('Optimizer is {}.'.format(optimizer))
    return model, optimizer, start_epoch, start_loss, load_amp


def save_model(path, epoch, train_loss, model, optimizer=None, amp_state=None):
    from apex.parallel import DistributedDataParallel
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        state_dict = model.module.state_dict()  # remove prefix 'module.'
    else:
        state_dict = model.state_dict()
    print(f'Saving {model.__class__.__name__} state dict...')

    data = {'epoch': epoch,
            'train_loss': train_loss,
            'model_state_dict': state_dict}
    if optimizer is not None:
        print(f'Saving {optimizer.__class__.__name__} state dict...')
        data['optimizer_state_dict'] = optimizer.state_dict()
    if amp_state is not None:
        print(f'Apex is used, saving all loss_scalers and their corresponding unskipped steps...')
        data['amp'] = amp_state
    torch.save(data, path)
    print(f'Checkpoint has been saved at {path}')


def initialize_weights(model):
    """Initialize model randomly.

    Args:
        model (nn.Module): input Pytorch model

    Returns:
        initialized model

    """
    # trick: obtain the class name of this current instance
    print(f"Initialize the weights of {model.__class__.__name__}.")
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.001)
            if m.bias is not None:  # bias are not used when we use BN layers
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            # torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    return model


class NetworkWrapper(torch.nn.Module):
    """Wrap the basenet and headnets into a single module."""

    def __init__(self, basenet, headnets):
        super(NetworkWrapper, self).__init__()
        self.basenet = basenet
        # Notice!  subnets in list or dict must be warped
        #   by ModuleList to register trainable params
        self.headnets = torch.nn.ModuleList(headnets)
        self.head_strides = [hn.stride for hn in headnets]
        self.head_names = [hn.head_name for hn in headnets]
        LOG.debug('warp the basnet and headnets into a whole model')

    def forward(self, img_tensor):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        feature_tuple = self.basenet(img_tensor)
        head_outputs = [hn(feature_tuple) for hn in self.headnets]
        LOG.debug('final output length of the model: %s ', len(head_outputs))
        return head_outputs


def basenet_factory(basenet_name):
    """
    Args:
        basenet_name:

    Returns:
        tuple: BaseNetwork, n_stacks, stride, max_stride, oup_dim

    """
    assert basenet_name in ['hourglass104', 'hourglass4stage'], \
        f'{basenet_name} is not implemented.'

    if 'hourglass104' in basenet_name:
        model = Hourglass104(None, 2)
        return model, 2, 4, 128, 256

    if 'hourglass52' in basenet_name:
        model = Hourglass104(None, 1)
        return model, 1, 4, 64, 256

    if 'hourglass4stage' in basenet_name:
        class IMHNOpt:
            nstack = 4  # stacked number of hourglass
            hourglass_inp_dim = 256
            increase = 128  # increased channels once down-sampling through networks

        net_opt = IMHNOpt()
        out_dim = 50
        raise Exception('unknown base network in {}'.format(basenet_name))
