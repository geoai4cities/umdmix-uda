import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree


def get_soft(t_vector, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    return t_soft


def get_kitti_soft(t_vector, labels, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    searched_idx = torch.logical_or(labels == 6, labels == 1)
    if searched_idx.sum() > 0:
        t_soft[searched_idx, 1] = max_val / 2
        t_soft[searched_idx, 6] = max_val / 2

    return t_soft


class SoftDICELoss(nn.Module):

    def __init__(
        self,
        ignore_label=None,
        powerize=True,
        use_tmask=True,
        neg_range=False,
        eps=0.05,
        is_kitti=False,
    ):
        super(SoftDICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.neg_range = neg_range
        self.eps = eps
        self.is_kitti = is_kitti

    def forward(self, output, target, return_class=False, is_kitti=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target_onehot = F.one_hot(target, num_classes=output.shape[1])
        if not self.is_kitti and not is_kitti:
            target_soft = get_soft(target_onehot, eps=self.eps)
        else:
            target_soft = get_kitti_soft(target_onehot, target, eps=self.eps)

        output = F.softmax(output, dim=-1)

        intersection = (output * target_soft).sum(dim=0)

        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target_onehot.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target_onehot.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)
        iou_class = tmask * 2 * intersection / union

        if self.neg_range:
            dice_loss = -iou.mean()
            dice_class = -iou_class
        else:
            dice_loss = 1 - iou.mean()
            dice_class = 1 - iou_class
        if return_class:
            return dice_loss.to(input_device), dice_class
        else:
            return dice_loss.to(input_device)


def get_neigbors_idx(sparse_tensor):
    coords = sparse_tensor.coordinates.cpu().numpy()
    tree = cKDTree(coords)
    _, idx = tree.query(coords, 5)
    return idx


class PosAwareLoss(nn.Module):
    def __init__(self, ign_label=-1, eps=1e-6):
        super(PosAwareLoss, self).__init__()
        # self.preds = preds
        # self.labels = labels
        # self.idx = idx
        self.ignore_label = ign_label
        self.eps = eps

    def CalcNdiff(self, labels, idx):
        Ndiff_list = []
        for lidx in range(len(labels)):
            curr_label = labels[lidx]
            neighbor_idx = idx[lidx]
            neighbor_labels = [labels[idx] for idx in neighbor_idx]
            Ndiff = sum(1 for label in neighbor_labels if label != curr_label)
            if Ndiff == 0:
                Ndiff = 1
            Ndiff_list.append(Ndiff)
        return Ndiff_list

    def forward(self, preds, labels, idx):
        Ndiff_list = torch.tensor(self.CalcNdiff(labels, idx), dtype=torch.float32)
        ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_label, reduction="none"
        )
        ce = ce_loss(preds, labels.long())
        # print("ce loss and ndiff", ce, Ndiff_list)
        wce = ce * Ndiff_list.to(preds.device)
        wce_mean = wce.mean()
        return wce_mean


class ShEntropy(nn.Module):
    def __init__(self, eps=1e-6):
        super(ShEntropy, self).__init__()
        self.eps = eps

    def forward(self, preds):
        preds = preds.float()
        # Apply softmax to get probabilities if not already probabilities
        probs = torch.softmax(preds, dim=-1)
        # Avoid log(0) by adding a small epsilon value
        probs = probs.clamp(min=self.eps)
        # Compute Shannon entropy
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy.mean()


class KLDiv(nn.Module):
    def __init__(self, eps=1e-6):
        super(KLDiv, self).__init__()
        self.eps = eps
        self.kldiv_loss = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, teacher_preds, student_preds):
        teacher_preds = teacher_preds.float()
        student_preds = student_preds.float()
        # teacher_preds should be log probabilities, so apply log_softmax
        teacher_log_probs = torch.log_softmax(teacher_preds, dim=-1)
        # student_preds should be probabilities, so apply softmax
        student_probs = torch.softmax(student_preds, dim=-1)
        # Compute KL divergence
        kldiv = self.kldiv_loss(teacher_log_probs, student_probs)
        return kldiv
