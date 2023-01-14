import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, outputs, targets):
        K = outputs.size()[-1]
        log_q = F.log_softmax(outputs, dim=-1)
        if self.reduction == 'sum':
            sigma_log_q = -log_q.sum()
        else:
            sigma_log_q = -log_q.sum(dim=-1)
            if self.reduction == 'mean':
                sigma_log_q = sigma_log_q.mean()

        loss = self.epsilon * sigma_log_q / K + (1 - self.epsilon) * F.nll_loss(log_q, targets,
                                                                                reduction=self.reduction)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criterion = LabelSmoothingCrossEntropy()
    out = torch.randn(20, 10)
    lbs = torch.randint(10, (20,))
    print('out:', out, out.size())
    print('lbs:', lbs, lbs.size())

    loss = criterion(out, lbs)
    print('loss:', loss)