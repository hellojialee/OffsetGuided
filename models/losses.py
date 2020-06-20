import logging
import torch


LOG = logging.getLogger(__name__)


class MultiTaskLoss(torch.nn.Module):

    def __init__(self, nstack_weights):
        super(MultiTaskLoss, self).__init__()
        self.nstack_weights = nstack_weights


class MultiHeadLoss(torch.nn.Module):
    def __init__(self, losses, lambdas):
        super(MultiHeadLoss, self).__init__()

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        # self.field_names = ['pif.c', 'pif.vec1', 'pif.scales1', 'paf.c', 'paf.vec1', 'paf.vec2']
        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, flat_head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, flat_head_losses