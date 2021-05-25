import numpy as np


def boolean_string(s):
    """Enable {'False', 'True'} string args in command line"""
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(learning_rate, world_size, optimizer,
                         epoch, step, len_epoch, use_warmup=False):
    factor = epoch // 15

    if epoch >= 60:
        factor = (epoch - 60) // 5

    lr = learning_rate * world_size * (0.2 ** factor)

    if epoch < 60:  # FIXME: change the LR schedule
        lr = learning_rate * world_size


    """Warmup the learning rate"""
    if use_warmup:
        if epoch < 15:
            # print('=============>  Using warm-up learning rate....')
            lr = lr * float(1 + step + epoch * len_epoch) / (
                    15. * len_epoch)  # len_epoch=len(train_loader)

    if 78 <= epoch < 92:
        lr = learning_rate * world_size

    if 92 <= epoch < 105:
        lr = 0.1 * learning_rate * world_size

    if 105 <= epoch < 110:
        lr = 0.01 * learning_rate * world_size

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_cyclic(optimizer, current_epoch, start_epoch,
                                swa_freqent=5, lr_max=4e-5, lr_min=2e-5):
    epoch = current_epoch - start_epoch

    lr = lr_max - (lr_max - lr_min) / (swa_freqent - 1) * (
            epoch - epoch // swa_freqent * swa_freqent)
    lr = round(lr, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
