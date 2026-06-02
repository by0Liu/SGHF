import os
import sys
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Import local modules
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.dirname(CURRENT_DIR)
if STAGE1_DIR not in sys.path:
    sys.path.insert(0, STAGE1_DIR)

# If your folder is named "models" instead of "model", change this import.
try:
    from model.gazefpn import GazeFPNPlus, gaze_loss
except ModuleNotFoundError:
    from models.gazefpn import GazeFPNPlus, gaze_loss

from dataset import MIMICDataset


# -----------------------------------------------------------------------------
# User settings: modify these paths before training
# -----------------------------------------------------------------------------
DATA_ROOT = r"/path/to/your/cxr_dataset_root"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VALID_CSV = os.path.join(DATA_ROOT, "valid.csv")
EYE_CAM_DIR = os.path.join(DATA_ROOT, "eye_cam")
SAVE_DIR = "./checkpoints"

NUM_CLASSES = 14
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 8
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 0

BACKBONE = "resnet101"      # "resnet50" or "resnet101"
FPN_OUT = 512
HEAD_MID = 256
USE_ASPP = True
TEMP = 1.0
FREEZE_BACKBONE = True       # keep the ImageNet backbone frozen during GazeFPN training

W_KL = 1.0
W_CC = 1.0
GAZE_SELECTION_LAMBDA = 1.0  # GazeScore = KL + lambda * (1 - CC)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Freeze or unfreeze the ResNet backbone inside GazeFPNPlus."""
    for module in [model.stem, model.layer1, model.layer2, model.layer3, model.layer4]:
        for param in module.parameters():
            param.requires_grad = requires_grad


def build_transforms(image_size: int = 512):
    return transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def compute_aucs(targets_t: torch.Tensor, preds_t: torch.Tensor) -> List[float]:
    """Compute per-class AUROC. Classes with only one label value return NaN."""
    try:
        from sklearn.metrics import roc_auc_score
        targets = targets_t.detach().cpu().numpy()
        preds = preds_t.detach().cpu().numpy()
        aucs = []
        for c in range(targets.shape[1]):
            y_true = targets[:, c]
            y_score = preds[:, c]
            if y_true.max() == y_true.min():
                aucs.append(float("nan"))
            else:
                aucs.append(float(roc_auc_score(y_true, y_score)))
        return aucs
    except Exception as exc:
        print(f"[WARN] AUC computation failed: {exc}")
        return [float("nan")] * preds_t.shape[1]


def load_eyegaze_map(file_rel_path: str, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Load one gaze map and normalize it as a probability map.

    Expected path:
        EYE_CAM_DIR / file_rel_path / eye_cam.npy

    The npy file is expected to be a dict containing key 'np_image'.
    """
    path = os.path.join(EYE_CAM_DIR, file_rel_path, "eye_cam.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gaze map not found: {path}")

    eye_obj = np.load(path, allow_pickle=True)
    if isinstance(eye_obj.item(), dict):
        eye_map = eye_obj.item()["np_image"]
    else:
        raise ValueError(f"Unexpected gaze npy format: {path}")

    H, W = out_hw
    eye_map = cv2.resize(eye_map, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    eye_map = eye_map - eye_map.min()

    max_val = float(eye_map.max())
    if max_val > 0:
        eye_map = eye_map / max_val

    sum_val = float(eye_map.sum())
    if sum_val <= 1e-8:
        eye_map[:] = 1.0 / (H * W)
    else:
        eye_map = eye_map / sum_val

    return torch.from_numpy(eye_map)[None, ...]  # [1, H, W]


def average_running(running: Dict[str, float], steps: int) -> Dict[str, float]:
    return {k: v / max(1, steps) for k, v in running.items()}


# -----------------------------------------------------------------------------
# Train / validation loops
# -----------------------------------------------------------------------------
def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[Dict[str, float], List[float], int]:
    model.train()
    bce_loss = nn.BCEWithLogitsLoss()

    running = {"gaze": 0.0, "cls": 0.0, "total": 0.0, "kl": 0.0, "cc": 0.0}
    targets_all, preds_all = [], []
    pbar = tqdm(dataloader, file=sys.stdout)

    for step, (images, labels, file_paths) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        B, _, H, W = images.shape

        q_g, _, y_logits = model(images)

        p_batch = torch.stack(
            [load_eyegaze_map(file_paths[i], (H, W)) for i in range(B)],
            dim=0
        ).to(device, non_blocking=True)

        loss_gaze, loss_kl, cc = gaze_loss(q_g, p_batch, w_kl=W_KL, w_cc=W_CC)
        loss_cls = bce_loss(y_logits, labels)
        loss = loss_gaze + loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running["gaze"] += float(loss_gaze.item())
        running["cls"] += float(loss_cls.item())
        running["total"] += float(loss.item())
        running["kl"] += float(loss_kl.item())
        running["cc"] += float(cc.item())

        targets_all.append(labels.detach().cpu())
        preds_all.append(torch.sigmoid(y_logits.detach()).cpu())

        avg = average_running(running, step)
        pbar.set_description(
            f"train: total {avg['total']:.3f} | gaze {avg['gaze']:.3f} | "
            f"cls {avg['cls']:.3f} | KL {avg['kl']:.3f} | CC {avg['cc']:.3f}"
        )

    targets_all = torch.cat(targets_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    aucs = compute_aucs(targets_all, preds_all)
    return running, aucs, len(dataloader)


@torch.no_grad()
def validate_one_epoch(model: nn.Module,
                       dataloader: DataLoader,
                       device: torch.device) -> Tuple[Dict[str, float], List[float], int]:
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()

    running = {"gaze": 0.0, "cls": 0.0, "total": 0.0, "kl": 0.0, "cc": 0.0}
    targets_all, preds_all = [], []
    pbar = tqdm(dataloader, file=sys.stdout)

    for step, (images, labels, file_paths) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        B, _, H, W = images.shape

        q_g, _, y_logits = model(images)
        p_batch = torch.stack(
            [load_eyegaze_map(file_paths[i], (H, W)) for i in range(B)],
            dim=0
        ).to(device, non_blocking=True)

        loss_gaze, loss_kl, cc = gaze_loss(q_g, p_batch, w_kl=W_KL, w_cc=W_CC)
        loss_cls = bce_loss(y_logits, labels)
        loss = loss_gaze + loss_cls

        running["gaze"] += float(loss_gaze.item())
        running["cls"] += float(loss_cls.item())
        running["total"] += float(loss.item())
        running["kl"] += float(loss_kl.item())
        running["cc"] += float(cc.item())

        targets_all.append(labels.detach().cpu())
        preds_all.append(torch.sigmoid(y_logits.detach()).cpu())

        avg = average_running(running, step)
        pbar.set_description(
            f"valid: total {avg['total']:.3f} | gaze {avg['gaze']:.3f} | "
            f"cls {avg['cls']:.3f} | KL {avg['kl']:.3f} | CC {avg['cc']:.3f}"
        )

    targets_all = torch.cat(targets_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    aucs = compute_aucs(targets_all, preds_all)
    return running, aucs, len(dataloader)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"TRAIN_CSV does not exist: {TRAIN_CSV}")
    if not os.path.exists(VALID_CSV):
        raise FileNotFoundError(f"VALID_CSV does not exist: {VALID_CSV}")
    if not os.path.exists(EYE_CAM_DIR):
        raise FileNotFoundError(f"EYE_CAM_DIR does not exist: {EYE_CAM_DIR}")

    transform = build_transforms(IMAGE_SIZE)
    train_set = MIMICDataset(TRAIN_CSV, transform)
    valid_set = MIMICDataset(VALID_CSV, transform)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print(f"[INFO] Training samples: {len(train_set)}")
    print(f"[INFO] Validation samples: {len(valid_set)}")

    model = GazeFPNPlus(
        num_classes=NUM_CLASSES,
        backbone=BACKBONE,
        temp=TEMP,
        fpn_out=FPN_OUT,
        head_mid=HEAD_MID,
        use_aspp=USE_ASPP,
    ).to(device)

    if FREEZE_BACKBONE:
        set_backbone_requires_grad(model, False)
        print("[INFO] Backbone is frozen.")
    else:
        print("[INFO] Backbone is trainable.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_total = float("inf")
    best_val_auc = -1.0
    best_gaze_score = float("inf")

    save_total = os.path.join(SAVE_DIR, "gaze_fpnv2_best_total.pth")
    save_auc = os.path.join(SAVE_DIR, "gaze_fpnv2_best_auc.pth")
    save_gaze = os.path.join(SAVE_DIR, "gaze_fpnv2_best_gaze.pth")

    for epoch in range(1, EPOCHS + 1):
        tr_stats, tr_aucs, tr_steps = train_one_epoch(model, train_loader, optimizer, device)
        va_stats, va_aucs, va_steps = validate_one_epoch(model, valid_loader, device)

        tr_avg = average_running(tr_stats, tr_steps)
        va_avg = average_running(va_stats, va_steps)
        va_auc_mean = float(np.nanmean(va_aucs))
        gaze_score = va_avg["kl"] + GAZE_SELECTION_LAMBDA * (1.0 - max(0.0, min(1.0, va_avg["cc"])))

        print(
            f"[Epoch {epoch:03d}/{EPOCHS}] "
            f"Train total {tr_avg['total']:.4f} | "
            f"Valid total {va_avg['total']:.4f} | "
            f"Valid mean AUC {va_auc_mean:.4f} | "
            f"KL {va_avg['kl']:.4f} | CC {va_avg['cc']:.4f} | "
            f"GazeScore {gaze_score:.4f}"
        )

        if va_avg["total"] < best_val_total:
            best_val_total = va_avg["total"]
            torch.save(model.state_dict(), save_total)
            print(f"  -> Save best(total) to {save_total}")

        if va_auc_mean > best_val_auc:
            best_val_auc = va_auc_mean
            torch.save(model.state_dict(), save_auc)
            print(f"  -> Save best(AUC) to {save_auc}")

        if gaze_score < best_gaze_score:
            best_gaze_score = gaze_score
            torch.save(model.state_dict(), save_gaze)
            print(f"  -> Save best(GazeScore) to {save_gaze}")

    print("[INFO] Finished training GazeFPN on CXR gaze data.")


if __name__ == "__main__":
    main()
