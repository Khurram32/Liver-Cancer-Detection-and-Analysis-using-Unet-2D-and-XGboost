"""
Section 3.2.3 / Table 4 — U-Net Training Protocol

Trains U-Net 2D with Adam (lr=1e-4), batch size 16, up to 50 epochs,
early stopping (patience=10) on validation loss, using the hybrid
Dice+BCE loss (Eq. 8).
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from unet import UNet2D
from losses import HybridDiceBCELoss
from config import unet_train_cfg
from data import SyntheticLiverCTDataset, LiTSVolumeDataset
from metrics_segmentation import dice_coefficient


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, loss_fn, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    dice_per_class = {c: [] for c in range(num_classes)}
    n_batches = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            for c in range(num_classes):
                for b in range(preds.shape[0]):
                    pred_c = (preds[b] == c)
                    gt_c = (masks_np[b] == c)
                    if gt_c.sum() == 0 and pred_c.sum() == 0:
                        continue
                    dice_per_class[c].append(dice_coefficient(pred_c, gt_c))

    avg_loss = total_loss / max(n_batches, 1)
    avg_dice = {c: (float(np.mean(v)) if len(v) > 0 else 0.0) for c, v in dice_per_class.items()}
    return avg_loss, avg_dice


def train(data_source: str = "synthetic", data_dir: str = None,
          n_samples: int = 70, epochs: int = None, batch_size: int = None,
          checkpoint_path: str = "../checkpoints/unet_best.pt", quick_test: bool = False):
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    n_epochs = unet_train_cfg.epochs if epochs is None else epochs
    bs = unet_train_cfg.batch_size if batch_size is None else batch_size
    if quick_test:
        n_epochs = min(n_epochs, 2)
        bs = min(bs, 2)

    # ---------------- Dataset ----------------
    if data_source == "lits" and data_dir is not None:
        full_dataset = LiTSVolumeDataset(data_dir, augment=True)
    else:
        full_dataset = SyntheticLiverCTDataset(n_samples=n_samples, augment=True)

    n_val = max(1, int(0.2 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    # ---------------- Model / Optim ----------------
    model = UNet2D().to(device)
    loss_fn = HybridDiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=unet_train_cfg.learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_total = 0.0
        n_batches = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            n_batches += 1

        train_loss = train_loss_total / max(n_batches, 1)
        val_loss, val_dice = evaluate(model, val_loader, loss_fn, device,
                                       unet_train_cfg.num_classes)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{n_epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | liver_dice={val_dice.get(1, 0):.4f} | "
              f"tumor_dice={val_dice.get(2, 0):.4f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "liver_dice": val_dice.get(1, 0), "tumor_dice": val_dice.get(2, 0),
        })

        # Early stopping (Table 4: patience=10)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= unet_train_cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(patience={unet_train_cfg.early_stopping_patience}).")
                break

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=["synthetic", "lits"], default="synthetic")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--checkpoint", default="../checkpoints/unet_best.pt")
    args = parser.parse_args()

    train(
        data_source=args.data_source,
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        quick_test=args.quick_test,
    )
