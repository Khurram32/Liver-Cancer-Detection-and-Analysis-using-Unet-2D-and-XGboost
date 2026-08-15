"""
Section 3.2.3 — Training Configuration and Optimization

Hybrid loss: L_total = lambda * L_Dice + (1 - lambda) * L_BCE   (Eq. 8)
L_Dice = 1 - (2*sum(p_i*g_i) + eps) / (sum(p_i) + sum(g_i) + eps)   (Eq. 9)
L_BCE  = -(1/N) * sum[ g_i*log(p_i) + (1-g_i)*log(1-p_i) ]          (Eq. 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import unet_train_cfg


class DiceLoss(nn.Module):
    """Eq. (9), computed per class and averaged (multi-class soft Dice)."""

    def __init__(self, num_classes: int = None, eps: float = None):
        super().__init__()
        self.num_classes = unet_train_cfg.num_classes if num_classes is None else num_classes
        self.eps = unet_train_cfg.dice_smooth_eps if eps is None else eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); targets: (B, H, W) with integer class labels
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # (B,H,W,C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()        # (B,C,H,W)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs, dim=dims) + torch.sum(targets_onehot, dim=dims)

        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss = 1.0 - dice_per_class.mean()
        return loss


class BCELossWrapper(nn.Module):
    """Eq. (10), implemented as multi-class cross-entropy (pixel-wise)."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class HybridDiceBCELoss(nn.Module):
    """Eq. (8): L_total = lambda * L_Dice + (1 - lambda) * L_BCE."""

    def __init__(self, lam: float = None, num_classes: int = None):
        super().__init__()
        self.lam = unet_train_cfg.dice_bce_lambda if lam is None else lam
        self.dice = DiceLoss(num_classes=num_classes)
        self.bce = BCELossWrapper()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_dice = self.dice(logits, targets)
        l_bce = self.bce(logits, targets)
        return self.lam * l_dice + (1 - self.lam) * l_bce


if __name__ == "__main__":
    logits = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 3, (2, 64, 64))
    loss_fn = HybridDiceBCELoss()
    loss = loss_fn(logits, targets)
    print("Hybrid loss:", loss.item())
