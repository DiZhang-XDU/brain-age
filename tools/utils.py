from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.nn import CrossEntropyLoss as CE
import torch
from torch.nn import MSELoss as MSE
from .losses import FocalLoss, FocalLoss_L2, dice_loss, CE_dice, CE_L1, DecadeCELoss, HuberLoss

def criterions(cfg):
    criterion_type = cfg.criterion
    # labelSmooth, softTarget, CE, tinyCE, focal, dice
    if criterion_type == 'labelSmooth':
        criterion = LabelSmoothingCrossEntropy(0.1)
    elif criterion_type == 'softTarget':
        criterion = SoftTargetCrossEntropy()
    elif criterion_type == 'CE':
        criterion = CE(ignore_index=5)
    elif criterion_type == 'CE+L1':
        criterion = CE_L1(ignore_index=5, alpha=0.1)
    elif criterion_type == 'tinyCE':
        criterion = CE(weight=torch.FloatTensor([1, 1.5, 1, 1, 1, 0]))
    elif criterion_type == 'focal':
        criterion = FocalLoss(weight = torch.Tensor([1,5]), alpha=0.25, gamma=2)
    elif criterion_type == 'focal+l2':
        criterion = FocalLoss_L2(weight = torch.Tensor([1,5]), alpha=0.25, gamma=2)
    elif criterion_type == 'dice':
        criterion = dice_loss()
    elif criterion_type == 'CE+dice':
        criterion = CE_dice(weight = torch.FloatTensor([1, 1]))
    elif criterion_type == 'DecadeCE':
        criterion = DecadeCELoss(sigma = 5)
    elif criterion_type == 'huber':
        criterion = HuberLoss()
    elif criterion_type == 'MSE':
        criterion = MSE()
    else:
        raise NotImplementedError(f"Unkown criterion: {criterion_type}")
    return criterion

def datasets(cfg):
    if cfg.task == '5min':
        from .dataset_age_10ep import XY_cohort_seq
        return XY_cohort_seq
    return None
