import torch.nn as nn
import torch.nn.functional as F
from utils.label_smoothing import LabelSmoothingCrossEntropy


def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha, label_smoothing):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities!
    """

    if label_smoothing:
        classification_loss = LabelSmoothingCrossEntropy()(outputs, labels)
    else:
        classification_loss = F.cross_entropy(outputs, labels)
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              classification_loss * (1. - alpha)

    return KD_loss
