# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)   # 修改
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum    # 修改
#  修改
#         iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
#         b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes[fg_mask].chunk(4, -1)
#         b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes[fg_mask].chunk(4, -1)
#         BX_L2Norm = torch.pow((b1_x1 - b2_x1), 2)
#         BY_L2Norm = torch.pow((b1_y1 - b2_y1), 2)
#         p1 = BX_L2Norm + BY_L2Norm
#         w_FroNorm = torch.pow((b1_x2 - b2_x2)/2, 2)
#         h_FroNorm = torch.pow((b1_y2 - b2_y2)/2, 2)
#         p2 = w_FroNorm + h_FroNorm
#         wasserstein = torch.exp(-torch.pow((p1+p2), 1 / 2) / 2.5)
#         wdloss = False    # 设置为 True 使用Normalized Gaussian Wasserstein Distance 设置 False 则用v8默认的
#         if wdloss:
#             loss_iou = (0.0 * ((1.0 - iou) * weight).sum() + 1.0 * ((1.0 - wasserstein) * weight).sum()) / target_scores_sum
#         else:
#             loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum



        # if type(iou) is tuple:
        #     if len(iou) == 2:
        #         loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
        #     else:
        #         loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum
        # else:
        #     loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)
