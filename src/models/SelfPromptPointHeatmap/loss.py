import torch.nn as nn

BASE_LOSS_DICT = dict(
    mse_loss=0,
    weight_dict=dict(
        heatmap_loss0 = 0.4,
        heatmap_loss1 = 0.6,
        heatmap_loss2 = 0.8,
        heatmap_loss3 = 1
    )
)

# BASE_LOSS_DICT = dict(
#     mse_loss=0,
#     weight_dict=dict(
#         heatmap_loss0 = 0.2,
#         heatmap_loss1 = 0.8,
#     )
# )

# BASE_LOSS_DICT = dict(
#     mse_loss=0,
#     weight_dict=dict(
#         heatmap_loss0 = 1
#     )
# )

class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()
        self.reduction = "mean"
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, pred, target):
        loss = self.mse_loss(pred, target)
        return loss / pred.size(0) if self.reduction == 'sum' else loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.func_list = [MseLoss(),]

    def forward(self, out_list):
        loss_dict = out_list[-1]
        out_dict = dict()
        weight_dict = dict()
        for key, item in loss_dict.items():
            out_dict[key] = self.func_list[int(item['type'].float().mean().item())](*item['params'])
            weight_dict[key] = item['weight'].mean().item()

        loss = 0.0
        for key in out_dict:
            loss += out_dict[key] * weight_dict[key]

        out_dict['loss'] = loss
        return out_dict
