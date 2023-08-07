import torch
import torch.nn as nn
import torch.nn.functional as F


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


class StructureLoss(nn.Module):
    def __init__(self,
                 weight=1.0):
        super(StructureLoss, self).__init__()
        self.weight = weight
        self.criterion = structure_loss

    def forward(self, pred, mask):
        loss = self.weight * self.criterion(pred, mask)
        return loss
