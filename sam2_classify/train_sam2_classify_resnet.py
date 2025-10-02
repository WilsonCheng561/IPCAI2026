#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline: ResNet-101 end-to-end classifier (no SAM2/Vision-Transformer).
- Keep the existing dataset/manifest loading, logging, metrics, confusion matrix saving.
- Minimal change to the training/eval loops — introduce head_type="resnet" and print shapes similarly.
- Checkpoints/log names renamed from *vit* to *resnet*.

Inputs:
  - Same manifest/label_map files as your ViT/SAM2 script
  - Images are read in BGR via OpenCV by FramePointDataset (unchanged)

Outputs:
  - Checkpoints in OUT_DIR/resnet101/
  - train_log_resnet.txt under SMALLFILE_ROOT
  - Confusion matrix PNGs saved in SMALLFILE_ROOT

Command example at bottom of file.
"""

import os, json, argparse, random, time, math, hashlib
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# torchvision for ResNet + image transforms
from torchvision import models, transforms

# ---- plotting (优雅降级) ----
HAS_MPL = True
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    HAS_MPL = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

# ---------- 路径 ----------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")
PRETRAIN_ROOT = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT = PRETRAIN_ROOT / "cwz" / "sam2_classifier"
DATASET_ROOT = Path("/projects/surgical-video-digital-twin/datasets/sam2_classifier")


# ---------- Utils ----------

def set_seed(seed: int = 42):
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_label_map(p: Path) -> Dict[str, Dict[str, int]]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class TrainingLogger:
    def __init__(self, filepath: Path):
        ensure_dir(filepath.parent)
        self.f = open(str(filepath), "a", buffering=1, encoding="utf-8")

    def write(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.f.write(f"[{ts}] {msg}\n")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# ---------- Class stats / balancing ----------

def _class_stats(train_csv: Path, label_map_json: Path):
    lm = load_label_map(label_map_json)
    tool2id = lm["tool_to_id"]
    df = pd.read_csv(train_csv)
    ids = [tool2id[t] for t in df["tool"] if t in tool2id]
    if len(ids) == 0:
        return None, None, None, None
    max_id = max(tool2id.values())
    counts = np.bincount(ids, minlength=max_id + 1).astype(np.float64)
    dist = {k: int(counts[v]) for k, v in tool2id.items()}
    nz = counts[counts > 0]
    imb_ratio = (counts.max() / nz.min()) if len(nz) else 1.0
    priors = counts / max(1.0, counts.sum())
    return counts, dist, float(imb_ratio), priors


def _ce_class_weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    w = 1.0 / np.log(1.0 + np.maximum(counts, 1.0))
    w = w * (len(w) / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def _sampling_weights_from_counts(counts: np.ndarray, alpha: float = 0.5, bg_factor: float = 0.3) -> np.ndarray:
    eps = 1e-12
    w = (np.maximum(counts, eps)) ** (-float(alpha))
    if len(w) > 0:
        w[0] *= float(bg_factor)
    return w


# ---------- Dataset (unchanged) ----------

class FramePointDataset(Dataset):
    def __init__(self, manifest_csv: Path, label_map_json: Path, resize: Optional[int] = None):
        super().__init__()
        self.df = pd.read_csv(manifest_csv)
        self.label_map = load_label_map(label_map_json)
        self.tool2id = self.label_map["tool_to_id"]
        self.resize = resize

        def has_points(s: str) -> bool:
            try:
                arr = json.loads(s) if isinstance(s, str) and s.strip() else []
                return len(arr) > 0
            except Exception:
                return False

        # 仍然沿用“有点才保留”的过滤，以保持与你原数据划分逻辑一致
        self.df = self.df[self.df["points_json"].apply(has_points)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_img(self, p: str):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        H0, W0 = img.shape[:2]
        if self.resize and self.resize > 0:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        return img, (H0, W0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img, (H0, W0) = self._load_img(row["image_path"])
        H1, W1 = img.shape[:2]

        # 保留 points 读取与尺度对齐，但后续模型不会再使用它们
        pts = json.loads(row["points_json"]) if isinstance(row["points_json"], str) and row[
            "points_json"].strip() else []
        pts_np = np.asarray(pts, np.float32) if len(pts) else np.zeros((0, 3), np.float32)
        if (H0 != H1) or (W0 != W1):
            sy = float(H1) / max(1.0, float(H0))
            sx = float(W1) / max(1.0, float(W0))
            if pts_np.size > 0:
                pts_np[:, 0] *= sx;
                pts_np[:, 1] *= sy

        pts_out = []
        for p in pts_np:
            if len(p) < 2: continue
            x = float(np.clip(p[0], 0, W1 - 1));
            y = float(np.clip(p[1], 0, H1 - 1))
            label = 1.0 if (len(p) >= 3 and float(p[2]) > 0) else 0.0
            pts_out.append([x, y, label])

        tool = str(row["tool"])
        tool_id = self.tool2id.get(tool, None)
        if tool_id is None:
            for k, v in self.tool2id.items():
                if str(k) == tool:
                    tool_id = v;
                    break
        if tool_id is None:
            raise KeyError(f"Tool '{tool}' not in label_map.json")

        return {
            "image": img,
            "points": np.array(pts_out, dtype=np.float32) if pts_out else np.zeros((0, 3), np.float32),
            "tool_id": int(tool_id),
            "meta": {
                "image_path": row["image_path"],
                "tool": tool,
                "frame_abs_index": int(row.get("frame_abs_index", -1)),
                "frame_idx_in_clip": int(row.get("frame_idx_in_clip", -1)),
                "clip_name": row.get("clip_name", ""),
                "task": row.get("task", ""),
                "orig_hw": (int(H0), int(W0)),
                "used_hw": (int(H1), int(W1)),
            }
        }


def collate_varlen(batch):
    images = [b["image"] for b in batch]
    points = [b["points"] for b in batch]
    targets = torch.tensor([b["tool_id"] for b in batch], dtype=torch.long)
    metas = [b["meta"] for b in batch]
    return {"images": images, "points": points, "targets": targets, "meta": metas}

    images = [b["image"] for b in batch]
    points = [b["points"] for b in batch]
    targets = torch.tensor([b["tool_id"] for b in batch], dtype=torch.long)
    metas = [b["meta"] for b in batch]
    return {"images": images, "points": points, "targets": targets, "meta": metas}


# ---------- ResNet-101 wrapper (extractor) ----------

class ResNetExtractor(nn.Module):
    """Takes a list of BGR np.ndarray images; returns a torch batch [B,3,H,W].
       The ResNet model (with classifier head) is kept separate as `head` to
       minimize changes in the training loop.
    """

    def __init__(self, device: str = "cuda", input_size: int = 224):
        super().__init__()
        self.device = device
        self.input_size = int(input_size)
        # ImageNet mean/std
        self.tx = transforms.Compose([
            transforms.ToTensor(),  # (H,W,C)->[0,1]
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def forward(self, images_bgr: List[np.ndarray], points_list: List[np.ndarray], metas: Optional[List[dict]] = None,
                return_4d: bool = True):
        # Convert BGR->RGB, resize if needed (dataset may have done it already)
        batch = []
        Ht = Wt = self.input_size
        for img_bgr in images_bgr:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[0] != Ht or img_rgb.shape[1] != Wt:
                img_rgb = cv2.resize(img_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
            t = self.tx(img_rgb)  # [3,H,W]
            batch.append(t)
        x = torch.stack(batch, dim=0).to(self.device, non_blocking=True)
        # For shape logging compatibility, we return a 4D tensor
        return x  # [B,3,H,W]


# ---------- Train / Eval helpers ----------

def _apply_logit_adjust(logits: torch.Tensor, log_prior: Optional[torch.Tensor], tau: float):
    if (log_prior is None) or (tau is None) or (tau <= 0):
        return logits
    return logits - float(tau) * log_prior.view(1, -1).to(logits.device)


# ---- 可视化：保存混淆矩阵图片 ----

def _save_confmat_figure(cm: np.ndarray, id2name: Dict[int, str], save_path: Path, title: str):
    if not HAS_MPL:
        print(f"[WARN] matplotlib not available, skip saving {save_path}")
        return
    ensure_dir(save_path.parent)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    im = ax.imshow(cm_norm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = [id2name.get(i, str(i)) for i in range(cm.shape[0])]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel="GT", xlabel="Pred", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), va="center", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


# ---------- Train / Eval ----------

def train_one_epoch(extractor: nn.Module, head: nn.Module, loader: DataLoader, device: str,
                    optimizer: torch.optim.Optimizer, epoch: int, logger: TrainingLogger, loss_fn,
                    max_grad_norm: float = 0.0, log_prior: Optional[torch.Tensor] = None,
                    logit_adjust_tau: float = 0.0, head_type: str = "resnet"):
    head.train()
    running_loss, n = 0.0, 0
    printed_shapes = False
    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar, 1):
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)
        with torch.no_grad():
            x4d = extractor(imgs, pts, metas, return_4d=True)  # [B,3,H,W]
        if not printed_shapes:
            B, C, H, W = x4d.shape
            out_dim = head.fc.out_features if hasattr(head, "fc") else head.classifier.out_features if hasattr(head,
                                                                                                               "classifier") else None
            logger.write(f"epoch={epoch} [shapes/train] x=[{B},{C},{H},{W}] -> logits=[{y.size(0)},{out_dim}]")
            print(f"[SHAPE][E{epoch}][train] ResNet in=({B},{C},{H},{W}) out=({y.size(0)},{out_dim})")
            printed_shapes = True
        logits = head(x4d)
        logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
        optimizer.step()
        bs = y.size(0)
        running_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        logger.write(f"epoch={epoch} step={step}/{len(loader)} train_loss={loss.item():.6f}")
    return running_loss / max(1, n)


@torch.no_grad()
def evaluate(extractor: nn.Module, head: nn.Module, loader: DataLoader, device: str,
             epoch: int, logger: TrainingLogger, loss_fn, split_name: str = "val",
             n_classes: Optional[int] = None, id2name: Optional[Dict[int, str]] = None,
             log_prior: Optional[torch.Tensor] = None, logit_adjust_tau: float = 0.0,
             head_type: str = "resnet"):
    torch.cuda.empty_cache()
    head.eval()
    total_loss, n = 0.0, 0
    correct = 0
    printed_shapes = False
    per_cls_total = None
    per_cls_correct = None
    cm = None
    if n_classes is not None:
        per_cls_total = torch.zeros(n_classes, dtype=torch.long)
        per_cls_correct = torch.zeros(n_classes, dtype=torch.long)
        cm = torch.zeros((n_classes, n_classes), dtype=torch.long)

    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch} [{split_name}]", leave=False)
    for batch in pbar:
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)
        x4d = extractor(imgs, pts, metas, return_4d=True)
        if not printed_shapes:
            B, C, H, W = x4d.shape
            out_dim = head.fc.out_features if hasattr(head, "fc") else head.classifier.out_features if hasattr(head,
                                                                                                               "classifier") else None
            print(f"[SHAPE][E{epoch}][{split_name}] ResNet in=({B},{C},{H},{W}) out=({y.size(0)},{out_dim})")
            logger.write(f"epoch={epoch} [shapes/{split_name}] x=[{B},{C},{H},{W}] -> logits=[{y.size(0)},{out_dim}]")
            printed_shapes = True
        logits = head(x4d)
        logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        if cm is not None:
            y_cpu = y.view(-1).to("cpu")
            p_cpu = preds.view(-1).to("cpu")
            idx = y_cpu * n_classes + p_cpu
            cm += torch.bincount(idx, minlength=n_classes * n_classes).view(n_classes, n_classes)
        if per_cls_total is not None:
            for c in range(n_classes):
                m = (y == c)
                if m.any():
                    per_cls_total[c] += int(m.sum().item())
                    per_cls_correct[c] += int((preds[m] == c).sum().item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)
    logger.write(f"epoch={epoch} {split_name}_loss={avg_loss:.6f} {split_name}_acc={acc:.6f}")
    print(f"[{split_name.upper()}] epoch {epoch} loss={avg_loss:.4f} acc={acc:.4f}")

    if (cm is not None) and (id2name is not None):
        cm_np = cm.numpy()
        tp = np.diag(cm_np)
        gt_per_cls = cm_np.sum(axis=1)
        pred_per_cls = cm_np.sum(axis=0)
        eps = 1e-12
        recall = np.divide(tp, gt_per_cls + eps)
        precision = np.divide(tp, pred_per_cls + eps)
        f1 = np.where((precision + recall) < eps, 0.0, 2 * precision * recall / (precision + recall))
        print(f"[{split_name.upper()} per-class metrics] (epoch {epoch})")
        for c in range(cm_np.shape[0]):
            name = id2name.get(c, f"class{c}")
            print(
                f" {c:2d} {name:>16s}: P={precision[c]:.4f} R={recall[c]:.4f} F1={f1[c]:.4f} (GT={int(gt_per_cls[c])}, Pred={int(pred_per_cls[c])})")
            logger.write(
                f"epoch={epoch} {split_name}_percls[{c}][{name}] P={precision[c]:.6f} R={recall[c]:.6f} F1={f1[c]:.6f} GT={int(gt_per_cls[c])} Pred={int(pred_per_cls[c])}")
        valid_gt = gt_per_cls > 0
        valid_pred = pred_per_cls > 0
        macro_recall = float(recall[valid_gt].mean()) if valid_gt.any() else float("nan")
        macro_precision = float(precision[valid_pred].mean()) if valid_pred.any() else float("nan")
        macro_f1 = 0.0 if (macro_precision + macro_recall) < eps else 2 * macro_precision * macro_recall / (
                    macro_precision + macro_recall)
        print(
            f"[{split_name.upper()} macro] (epoch {epoch}): P={macro_precision:.4f} R={macro_recall:.4f} F1={macro_f1:.4f}")
        logger.write(f"epoch={epoch} {split_name}_macro P={macro_precision:.6f} R={macro_recall:.6f} F1={macro_f1:.6f}")
        save_path = SMALLFILE_ROOT / f"cm_{split_name}_epoch{epoch:03d}.png"
        _save_confmat_figure(cm_np, id2name, save_path, title=f"Confusion Matrix [{split_name}] epoch {epoch}")
        print(f"[{split_name.upper()}] confusion matrix saved to: {save_path}")

    return avg_loss, acc


# ---------- Build ResNet-101 ----------

def build_resnet101(n_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        except Exception:
            model = models.resnet101(weights=None)
    else:
        model = models.resnet101(weights=None)
    # Replace final FC
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(SMALLFILE_ROOT))
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--label-map", type=str, default=None)

    # ResNet options
    ap.add_argument("--arch", choices=["resnet101"], default="resnet101")
    ap.add_argument("--input-size", type=int, default=224, help="input spatial size for ResNet")
    ap.add_argument("--pretrained", action="store_true", help="use ImageNet-pretrained weights")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--resize", type=int, default=224,
                    help="dataset-side resize before transforms; set None to keep original and let extractor resize")
    ap.add_argument("--out-dir", type=str, default=str(CKPT_ROOT))

    # balancing / focal / logit-adjust (kept)
    ap.add_argument("--balance", choices=["none", "weights", "sampler", "auto"], default="auto")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--reweight-alpha", type=float, default=1.0)
    ap.add_argument("--bg-factor", type=float, default=0.1)
    ap.add_argument("--logit-adjust", type=float, default=1.0)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--sched", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--max-grad-norm", type=float, default=0.0)
    ap.add_argument("--resume", type=str, default=None)

    # eval/test 独立 batch size
    ap.add_argument("--val-batch-size", type=int, default=None, help="eval/val loader bs; default=--batch-size")
    ap.add_argument("--test-batch-size", type=int, default=None, help="test loader bs; default=--batch-size")

    # bg 掩码策略参数保留（数据集里用到）

    args = ap.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = "cuda";
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else (data_root / "manifest.csv")
    label_map_path = Path(args.label_map) if args.label_map else (data_root / "label_map.json")

    out_dir = Path(args.out_dir) / args.arch
    ensure_dir(out_dir);
    ensure_dir(SMALLFILE_ROOT)

    # 使用现成 split
    train_csv = SMALLFILE_ROOT / "train_manifest_10.csv"
    val_csv = SMALLFILE_ROOT / "val_manifest_10.csv"
    test_csv = SMALLFILE_ROOT / "test_manifest_10.csv"
    if train_csv.exists() and val_csv.exists():
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        df_test = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame(columns=df_train.columns)
    else:
        raise FileNotFoundError("train/val manifest not found")

    # label_map check
    lm = load_label_map(label_map_path)
    tool2id = lm["tool_to_id"]
    assert "background" in tool2id and tool2id["background"] == 0
    id2name = {int(v): str(k) for k, v in tool2id.items()}

    # datasets & loaders
    ds_train = FramePointDataset(train_csv, label_map_path, resize=args.resize)
    ds_val = FramePointDataset(val_csv, label_map_path, resize=args.resize) if len(df_val) else None if len(
        df_val) else None
    ds_test = FramePointDataset(test_csv, label_map_path, resize=args.resize) if len(df_test) else None if len(
        df_test) else None

    train_bs = args.batch_size
    val_bs = args.val_batch_size if args.val_batch_size is not None else args.batch_size
    test_bs = args.test_batch_size if args.test_batch_size is not None else args.batch_size

    sampler = None
    counts, train_dist, imb_ratio, priors = _class_stats(train_csv, label_map_path)
    if train_dist is not None:
        print(f"[CHECK] train distribution per class = {train_dist} | imbalance ratio={imb_ratio:.2f}")
    class_weights_ce = _ce_class_weights_from_counts(counts) if counts is not None else None
    log_prior = torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32)) if priors is not None else None

    if (args.balance in ("auto", "sampler")) and (counts is not None):
        use_sampler = (args.balance == "sampler") or (args.balance == "auto" and imb_ratio >= 5.0)
        if use_sampler:
            per_class_sw = _sampling_weights_from_counts(counts, alpha=args.reweight_alpha, bg_factor=args.bg_factor)
            sample_ids = [tool2id.get(t, 0) for t in ds_train.df["tool"].tolist()]
            sw = [float(per_class_sw[c]) for c in sample_ids]
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
            print(f"[INFO] Using WeightedRandomSampler (alpha={args.reweight_alpha}, bg_factor={args.bg_factor}).")

    dl_train = DataLoader(ds_train, batch_size=train_bs, shuffle=(sampler is None), num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True, sampler=sampler)
    dl_val = DataLoader(ds_val, batch_size=val_bs, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_varlen, pin_memory=True) if ds_val else None
    dl_test = DataLoader(ds_test, batch_size=test_bs, shuffle=False, num_workers=args.workers,
                         collate_fn=collate_varlen, pin_memory=True) if ds_test else None

    # Build extractor + model
    extractor = ResNetExtractor(device=device, input_size=args.input_size)
    n_classes = len(tool2id)
    if args.arch == "resnet101":
        model = build_resnet101(n_classes=n_classes, pretrained=args.pretrained).to(device)
    else:
        raise NotImplementedError(args.arch)

    # Optimizer / scheduler / loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    using_sampler = sampler is not None
    if args.focal:
        gamma = 2.0
        alpha = (None if using_sampler else (class_weights_ce.to(device) if class_weights_ce is not None else None))

        def focal_loss(logits, target):
            ce = F.cross_entropy(logits, target, reduction="none", weight=alpha)
            pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0)
            fl = ((1 - pt) ** gamma) * ce
            return fl.mean()

        loss_fn = focal_loss
        print("[INFO] Using Focal Loss (gamma=2).")
    else:
        if (not using_sampler) and (args.balance in ("weights", "auto")) and (class_weights_ce is not None) and (
                imb_ratio is not None and imb_ratio >= 5.0):
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_ce.to(device))
            print("[INFO] Using CE with class weights.")
        else:
            loss_fn = nn.CrossEntropyLoss()

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.01) if args.sched == "cosine" else None

    # resume
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        try:
            model.load_state_dict(state, strict=True)
            print(f"[RESUME] Loaded model from {args.resume}")
        except Exception as e:
            print(f"[RESUME] Strict load failed: {e}\nTrying non-strict...")
            model.load_state_dict(state, strict=False)
            print(f"[RESUME] Non-strict loaded from {args.resume}")

    # logger
    log_file = SMALLFILE_ROOT / "train_log_resnet.txt"
    logger = TrainingLogger(log_file)

    try:
        best_acc = -1.0;
        best_epoch = -1
        patience_left = args.patience
        best_path = out_dir / "best_resnet101.pt"

        for epoch in range(1, args.epochs + 1):
            if args.warmup_epochs and epoch <= args.warmup_epochs:
                warmup_ratio = epoch / max(1, args.warmup_epochs)
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * (0.1 + 0.9 * warmup_ratio)

            tr_loss = train_one_epoch(extractor, model, dl_train, device, opt, epoch, logger,
                                      loss_fn=loss_fn, max_grad_norm=args.max_grad_norm,
                                      log_prior=log_prior, logit_adjust_tau=args.logit_adjust,
                                      head_type="resnet")

            va_loss, va_acc = evaluate(extractor, model, dl_val, device, epoch, logger, loss_fn=loss_fn,
                                       split_name="val", n_classes=n_classes, id2name=id2name,
                                       log_prior=log_prior, logit_adjust_tau=args.logit_adjust,
                                       head_type="resnet") if dl_val else (0.0, 0.0)

            print(f"[{epoch:02d}] train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} val_acc {va_acc:.3f}")
            if sched is not None and (not args.warmup_epochs or epoch > args.warmup_epochs):
                sched.step()

            improved = (dl_val is None) or (va_acc > best_acc)
            if improved:
                best_acc = va_acc;
                best_epoch = epoch;
                patience_left = args.patience
                torch.save({
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "n_classes": n_classes,
                    "tool_to_id": tool2id,
                    "args": vars(args)
                }, str(best_path))
                logger.write(f"epoch={epoch} SAVED best_resnet -> {best_path}")
            else:
                patience_left -= 1

            if (epoch % 5 == 0) or (epoch == args.epochs):
                ep_path = out_dir / f"resnet101_epoch{epoch:03d}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "n_classes": n_classes,
                    "tool_to_id": tool2id,
                    "args": vars(args)
                }, str(ep_path))
                logger.write(f"epoch={epoch} SAVED periodic_resnet -> {ep_path}")

            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best val acc={best_acc:.4f} (epoch {best_epoch}).")
                logger.write(f"early_stop best_acc={best_acc:.6f} best_epoch={best_epoch}")
                break

        if dl_test:
            test_loss, test_acc = evaluate(extractor, model, dl_test, device, epoch=best_epoch, logger=logger,
                                           loss_fn=loss_fn, split_name="test", n_classes=n_classes, id2name=id2name,
                                           log_prior=log_prior, logit_adjust_tau=args.logit_adjust, head_type="resnet")
            print(f"[TEST] loss {test_loss:.4f} acc {test_acc:.3f}")
            logger.write(f"test_loss={test_loss:.6f} test_acc={test_acc:.6f}")

        print(f"Done. Best val acc={best_acc:.4f} at epoch {best_epoch}. Saved to: {best_path}")
        logger.write(f"done best_acc={best_acc:.6f} best_epoch={best_epoch} path={best_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

# python /home/wcheng31/sam2_classify/train_sam2_classify_resnet.py \
#   --epochs 20 --batch-size 128 --val-batch-size 64 \
#   --seed 123 --patience 5 \
#   --input-size 224 --pretrained \
#   --balance sampler --focal --reweight-alpha 1.0 --bg-factor 0.1 --logit-adjust 1.0 \
#   --out-dir /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier

