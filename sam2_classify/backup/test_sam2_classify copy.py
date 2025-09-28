#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified tester for both training regimes:

  A) Frozen SAM2 + trained head   -> --resume-mode head
     (uses Sam2OfficialWrapper from your frozen training script)

  B) Finetuned SAM2 + head        -> --resume-mode full
     (uses FineTuneSam2Wrapper from your finetune script)

Features:
- prepared CSV or OOD(raw) builder (same as before)
- eval metrics: overall acc, per-class acc (+ optional top-k)
- apply logit-adjust in eval/vis if ckpt had it during training
- background mask mode at test time to match training (pos/global/mix)
- visualization: de-duplicate by image_path, shuffle, optional class-balanced sampling
- small memory footprint (no grads, AMP off by default, minimal tensors kept)

Author: Wenzheng Cheng
"""

import os, json, argparse, random, hashlib, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

# ---------- Paths ----------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"

# ---------- Import wrappers/heads from your training scripts ----------
import sys
sys.path.append(str(SMALLFILE_ROOT))

# try to import both wrappers; they live in two different train files in your env
Sam2OfficialWrapper = None
FineTuneSam2Wrapper = None
MLPHead = None
MLPBNHead = None
CosineClassifier = None

# Frozen training script (your last message)
try:
    from train_sam2_classify import Sam2OfficialWrapper as _FrozenWrapper
    from train_sam2_classify import MLPHead as _MLPHead
    try:
        from train_sam2_classify import MLPBNHead as _MLPBNHead, CosineClassifier as _CosCls
    except Exception:
        _MLPBNHead, _CosCls = None, None
    Sam2OfficialWrapper = _FrozenWrapper
    MLPHead = _MLPHead; MLPBNHead = _MLPBNHead; CosineClassifier = _CosCls
except Exception as e:
    print(f"[WARN] cannot import Sam2OfficialWrapper from frozen script: {e}")

# Finetune training script (the earlier e2e finetune one)
try:
    from train_sam2_classify_finetune import FineTuneSam2Wrapper as _FTWrapper
    if MLPHead is None:
        from train_sam2_classify_finetune import MLPHead as _MLPHead2
        MLPHead = _MLPHead2
    try:
        from train_sam2_classify_finetune import MLPBNHead as _MLPBNHead2, CosineClassifier as _CosCls2
        if MLPBNHead is None: MLPBNHead = _MLPBNHead2
        if CosineClassifier is None: CosineClassifier = _CosCls2
    except Exception:
        pass
    FineTuneSam2Wrapper = _FTWrapper
except Exception as e:
    print(f"[WARN] cannot import FineTuneSam2Wrapper from finetune script: {e}")

# ========================= Datasets =========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_label_map(p: Path) -> Dict[str, Dict[str, int]]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(obj, p: Path):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

TOOL_TO_ID_OOD = {"background":0, "clipper":1, "grasper":2, "hook":3, "scissors":4}
ID_TO_TOOL_OOD = {v:k for k,v in TOOL_TO_ID_OOD.items()}

def _map_obj_id_to_cls(obj_id: int) -> str:
    if obj_id in (2,3,4): return "grasper"
    if obj_id == 6:       return "hook"
    if obj_id == 7:       return "scissors"
    if obj_id == 8:       return "clipper"
    return "background"

def _build_ood_manifest(raw_root: Path, videos: List[str], out_csv: Path) -> pd.DataFrame:
    rows = []
    for vid in videos:
        vid_dir = raw_root / vid / "ws_0"
        img_dir = vid_dir / "images"
        prm = vid_dir / "prompts.json"
        if not prm.exists():
            print(f"[WARN] missing prompts: {prm}")
            continue
        data = _load_json(prm)
        for rec in data:
            frame_file = rec.get("frame_file", None)
            if not frame_file:
                fid = rec.get("frame_id", None)
                frame_file = f"{int(fid):07d}.jpg" if fid is not None else None
            if not frame_file:
                continue
            img_path = img_dir / frame_file
            if not img_path.exists():
                continue

            buckets: Dict[str, List[List[float]]] = {}
            for obj in rec.get("objects", []):
                oid = int(obj.get("obj_id", -1))
                cls = _map_obj_id_to_cls(oid)
                pts = obj.get("points", [])
                labs = obj.get("labels", [])
                if len(labs) != len(pts):
                    L = min(len(pts), len(labs))
                    pts, labs = pts[:L], labs[:L]
                for (xy, lab) in zip(pts, labs):
                    if not isinstance(xy, (list,tuple)) or len(xy) < 2:
                        continue
                    x, y = float(xy[0]), float(xy[1])
                    label = 1.0 if int(lab) > 0 else 0.0
                    buckets.setdefault(cls, []).append([x, y, label])

            for cls, pts in buckets.items():
                if len(pts) == 0: continue
                rows.append({
                    "image_path": str(img_path),
                    "tool": cls,
                    "points_json": json.dumps(pts),
                    "frame_id": rec.get("frame_id", -1),
                    "clip_name": vid,
                })

    df = pd.DataFrame(rows)
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    print(f"[BUILD][OOD] saved -> {out_csv}  (#rows={len(df)})")
    return df

class FramePointDataset(Dataset):
    """Light eval dataset (no bg-policy inside; we do it in the tester to unify both regimes)."""
    def __init__(self, manifest_csv: Path, label_map_json: Path, resize: Optional[int] = None):
        super().__init__()
        self.df = pd.read_csv(manifest_csv)
        with open(label_map_json, "r", encoding="utf-8") as f:
            lm = json.load(f)
        self.tool2id = lm["tool_to_id"]
        self.resize = resize

        def has_points(s: str) -> bool:
            try:
                arr = json.loads(s) if isinstance(s, str) and s.strip() else []
                return len(arr) > 0
            except Exception:
                return False
        self.df = self.df[self.df["points_json"].apply(has_points)].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def _load_img(self, p: str):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(p)
        if self.resize and self.resize > 0:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_img(row["image_path"])
        H, W = img.shape[:2]
        pts = json.loads(row["points_json"])
        pts_out = []
        for p in pts:
            x = float(np.clip(p[0], 0, W-1)); y = float(np.clip(p[1], 0, H-1))
            label = 1.0 if float(p[2]) > 0 else 0.0
            pts_out.append([x, y, label])
        tool = str(row["tool"])
        tool_id = int(self.tool2id[tool])
        return {
            "image": img,
            "points": np.array(pts_out, dtype=np.float32),
            "tool_id": tool_id,
            "meta": {"image_path": row["image_path"], "tool": tool, "clip_name": row.get("clip_name","")}
        }

def collate_varlen(batch):
    images  = [b["image"]  for b in batch]
    points  = [b["points"] for b in batch]
    targets = torch.tensor([b["tool_id"] for b in batch], dtype=torch.long)
    metas   = [b["meta"]   for b in batch]
    return {"images": images, "points": points, "targets": targets, "meta": metas}

# ========================= Heads =========================

def _parse_hidden_list(s: str) -> List[int]:
    out = []
    try:
        for p in str(s).split(","):
            p = p.strip()
            if p:
                out.append(int(p))
    except Exception:
        pass
    return out

def _build_head_from_ckpt_args(in_dim: int, n_classes: int, args_dict: Dict, override_head: Optional[str] = None):
    head_type = override_head or args_dict.get("head", "mlp")
    drop      = float(args_dict.get("drop", 0.0))
    scale     = float(args_dict.get("scale", 16.0))
    hidden    = args_dict.get("hidden", "0")

    if MLPHead is None:
        raise RuntimeError("MLPHead not available from training imports.")

    if head_type == "linear":
        return MLPHead(in_dim, n_classes, hidden=0, drop=drop)
    if head_type == "mlp":
        h = 0
        if isinstance(hidden, str):
            lst = _parse_hidden_list(hidden); h = (lst[0] if lst else 0)
        elif isinstance(hidden, (list, tuple)) and len(hidden) > 0:
            h = int(hidden[0])
        return MLPHead(in_dim, n_classes, hidden=h, drop=drop)
    if head_type == "mlp_bn" and MLPBNHead is not None:
        lst = _parse_hidden_list(hidden) or [1024, 512]
        return MLPBNHead(in_dim, n_classes, hidden_layers=lst, drop=drop)
    if head_type == "cosine" and CosineClassifier is not None:
        return CosineClassifier(in_dim, n_classes, scale=scale)
    return MLPHead(in_dim, n_classes, hidden=0, drop=drop)

# ========================= Eval Core =========================

def _apply_logit_adjust(logits: torch.Tensor, log_prior: Optional[torch.Tensor], tau: float):
    if (log_prior is None) or (tau is None) or (tau <= 0):
        return logits
    return logits - float(tau) * log_prior.view(1, -1).to(logits.device)

def _prep_points_for_bg(points: np.ndarray, tool_id: int, bg_mask_mode: str, bg_mix_p: float) -> np.ndarray:
    """For resume-mode=head: emulate training-time bg policy by editing points before calling extractor."""
    if tool_id != 0:
        return points
    mode = bg_mask_mode
    if mode == "mix":
        mode = "global" if (random.random() < float(bg_mix_p)) else "pos"
    if mode == "global":
        return np.zeros((0,3), np.float32)  # feed empty -> extractor falls back to global GAP
    # pos: set all labels to 1
    pts = np.asarray(points, np.float32).copy()
    if pts.size > 0:
        pts[:, 2] = 1.0
    return pts

@torch.no_grad()
def evaluate_batchwise(extractor, head: nn.Module, loader: DataLoader, device: str,
                       log_prior: Optional[torch.Tensor], tau: float,
                       bg_mask_mode: str, bg_mix_p: float, resume_mode: str,
                       topk_eval: int = 1):
    head.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_n, total_correct = 0.0, 0, 0
    y_true, y_pred = [], []
    n_classes = None

    pbar = tqdm(loader, total=len(loader), ncols=100, desc="[test] eval", leave=True)
    for batch in pbar:
        imgs, pts_list, y = batch["images"], batch["points"], batch["targets"].to(device)
        # (可选) 在 head-only 模式下对背景点进行策略对齐
        if resume_mode == "head":
            new_pts = []
            for p, t in zip(pts_list, batch["targets"].tolist()):
                new_pts.append(_prep_points_for_bg(p, int(t), bg_mask_mode, bg_mix_p))
            pts_list = new_pts

        feats = extractor(imgs, pts_list, batch.get("meta", None))
        logits = head(feats)
        logits = _apply_logit_adjust(logits, log_prior, tau)
        loss = ce(logits, y)

        pred = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        total_correct += (pred == y).sum().item()

        y_true.append(y.detach().cpu().numpy()); y_pred.append(pred.detach().cpu().numpy())
        if n_classes is None:
            n_classes = int(logits.shape[1])

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(total_correct/max(1,total_n)):.3f}")

    y_true = np.concatenate(y_true) if y_true else np.zeros((0,), np.int64)
    y_pred = np.concatenate(y_pred) if y_pred else np.zeros((0,), np.int64)
    overall_acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    per_class_acc, per_class_cnt = {}, {}
    if len(y_true):
        for c in np.unique(y_true):
            idx = (y_true == c)
            per_class_cnt[int(c)] = int(idx.sum())
            per_class_acc[int(c)] = float((y_pred[idx] == c).mean()) if idx.any() else 0.0

    avg_loss = total_loss / max(1, total_n)
    return avg_loss, overall_acc, per_class_acc, per_class_cnt, y_true, y_pred

# -------- Visualization helpers (dedupe + stratified sampling) --------

def _compute_resize_and_padding(H0, W0, H_in, W_in):
    """
    把原图(H0,W0)按最短边适配到SAM2输入(H_in,W_in)，返回:
      scale  - 统一缩放系数 (float)
      pad_x  - 左/右总padding的一半 (float)
      pad_y  - 上/下总padding的一半 (float)
      new_h, new_w - 缩放后的尺寸 (int)
    约定：采用常见的“等比缩放 + 居中对齐padding”策略。
    """
    scale = min(H_in / float(H0), W_in / float(W0))
    new_h, new_w = int(round(H0 * scale)), int(round(W0 * scale))
    pad_y = (H_in - new_h) * 0.5
    pad_x = (W_in - new_w) * 0.5
    return scale, pad_x, pad_y, new_h, new_w

def _map_points_to_sam_input(pts_np, H0, W0, H_in, W_in):
    """
    显式把原图坐标(x,y)映射到SAM2输入坐标系(考虑等比缩放+居中padding)。
    输入:
      pts_np: (N,3) [x,y,label]，原图坐标
    返回:
      coords: (1,N,2) torch.float32，SAM2输入坐标
      labels: (1,N)   torch.int64
    """
    pts_np = np.asarray(pts_np, np.float32)
    if pts_np.size == 0:
        return None, None

    scale, pad_x, pad_y, _, _ = _compute_resize_and_padding(H0, W0, H_in, W_in)

    xs = pts_np[:, 0] * scale + pad_x
    ys = pts_np[:, 1] * scale + pad_y

    # clamp到合法范围
    xs = np.clip(xs, 0, W_in - 1)
    ys = np.clip(ys, 0, H_in - 1)

    coords = np.stack([xs, ys], axis=1)[None, ...].astype(np.float32)  # (1,N,2)
    labels = (pts_np[:, 2] > 0).astype(np.int64)[None, ...]            # (1,N)

    coords = torch.from_numpy(coords)
    labels = torch.from_numpy(labels)
    return coords, labels

def _resize_mask_back_to_orig(mask_prob_2d, H0, W0, H_in, W_in):
    """
    把SAM输入尺寸上的mask按“先去padding→再等比反缩放→还原到原图”返回(H0,W0)大小。
    输入:
      mask_prob_2d: (H_in,W_in) float32
    """
    scale, pad_x, pad_y, new_h, new_w = _compute_resize_and_padding(H0, W0, H_in, W_in)

    # 1) 去掉padding（注意四边padding可能不是整数，这里做round并clamp）
    x0 = int(round(pad_x)); y0 = int(round(pad_y))
    x1 = int(round(pad_x + new_w)); y1 = int(round(pad_y + new_h))
    x0 = max(0, min(x0, W_in - 1)); x1 = max(1, min(x1, W_in))
    y0 = max(0, min(y0, H_in - 1)); y1 = max(1, min(y1, H_in))
    cropped = mask_prob_2d[y0:y1, x0:x1]

    # 2) 反缩放到原图
    mask_back = cv2.resize(cropped, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return mask_back




def _prepare_vis_df(df: pd.DataFrame, num_vis: int, stratified: bool) -> pd.DataFrame:
    # 1) drop duplicates by image_path
    if "image_path" in df.columns:
        df = df.drop_duplicates(subset=["image_path"]).copy()
    # 2) shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if not stratified or "tool" not in df.columns:
        return df.head(min(num_vis, len(df)))

    # 3) class-balanced sample
    k = df["tool"].nunique()
    per_cls = max(1, num_vis // k)
    parts = []
    for _, g in df.groupby("tool"):
        parts.append(g.head(min(per_cls, len(g))))
    df_bal = pd.concat(parts).sample(frac=1.0, random_state=123).reset_index(drop=True)
    return df_bal.head(min(num_vis, len(df_bal)))


@torch.no_grad()
def infer_and_classify_with_points(
    extractor,
    head,
    img_bgr,             # (H0,W0,3) uint8 (BGR)
    pts_np,              # (N,3) [x,y,label] in ORIGINAL image coords
    device="cuda",
    conf_thr=0.5,
    log_prior=None,
    tau=0.0,
    draw_points=True,
):
    """
    修复要点：
      1) 显式计算等比缩放+居中padding的坐标映射，避免(sy,sx)约定不一致导致的偏移；
      2) 掩码回原图：先crop掉padding，再resize回(H0,W0)。
    """
    H0, W0 = img_bgr.shape[:2]

    # --- 取SAM输入张量 & 尺寸 ---
    img_t, (h0, w0), (H_in, W_in), _, _ = extractor._preprocess_manual(img_bgr)
    img_feat, img_pe, high_res = extractor._get_image_embed(img_t)

    # --- 点映射到SAM输入坐标 ---
    pts_np = np.asarray(pts_np, np.float32) if pts_np is not None else np.zeros((0, 3), np.float32)
    if pts_np.shape[0] == 0:
        mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
        pts_draw = []
    else:
        coords, labels = _map_points_to_sam_input(pts_np, H0, W0, H_in, W_in)
        coords = coords.to(img_feat.device); labels = labels.to(img_feat.device)

        if int(labels.max().item()) <= 0:
            mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
        else:
            sp, dp = extractor._encode_prompts(coords, labels)
            logits_m = extractor._decode_mask(img_feat, img_pe, sp, dp, high_res)
            mask = torch.sigmoid(logits_m)
            if mask.shape[-2:] != img_feat.shape[-2:]:
                mask = F.interpolate(mask, size=img_feat.shape[-2:], mode="bilinear", align_corners=False)
            if (not torch.isfinite(mask).all()) or (mask.sum() <= 1e-5):
                mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)

        # 原图坐标用于画点
        pts_draw = [(float(x), float(y), int(lab > 0)) for (x, y, lab) in pts_np]

    # --- masked GAP 特征 ---
    feat = (img_feat * mask).flatten(2).sum(dim=-1) / (mask.flatten(2).sum(dim=-1) + 1e-6)
    feat = feat.squeeze(0)

    # --- 分类 + (可选)logit-adjust ---
    logits = head(feat.unsqueeze(0).to(device))
    if (log_prior is not None) and (tau is not None) and (tau > 0):
        logits = logits - float(tau) * log_prior.view(1, -1).to(logits.device)
    prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_id = int(prob.argmax())

    # --- 掩码回原图(精确去padding) ---
    mask2d_in = mask.squeeze().detach().float().cpu().numpy()  # (H_feat,W_feat) == (H_in/feat, W_in/feat) after upsample
    if mask2d_in.shape != (H_in, W_in):
        mask2d_in = cv2.resize(mask2d_in, (W_in, H_in), interpolation=cv2.INTER_LINEAR)
    mask2d_orig = _resize_mask_back_to_orig(mask2d_in, H0, W0, H_in, W_in)

    # --- 右图：黑遮罩 + 画点 ---
    vis_right = img_bgr.copy()
    m_bin = (mask2d_orig >= float(conf_thr))
    vis_right[m_bin] = 0
    if draw_points and len(pts_draw) > 0:
        for (x, y, is_pos) in pts_draw:
            c = (0, 255, 0) if is_pos else (0, 0, 255)
            cv2.circle(vis_right, (int(round(x)), int(round(y))), 5, c, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(vis_right, (int(round(x)), int(round(y))), 6, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return pred_id, prob, vis_right, mask2d_orig



@torch.no_grad()
def _mask_and_feat_for_one(
    extractor,
    img_bgr,
    pts_np,
    tool_id,
    resume_mode,
    bg_mask_mode,
    bg_mix_p,
):
    """
    与上面一致：显式坐标映射 + 去padding后再反缩放。
    """
    def _prep_bg_points(points):
        if tool_id != 0:
            return points
        mode = bg_mask_mode
        if mode == "mix":
            mode = "global" if (np.random.rand() < float(bg_mix_p)) else "pos"
        if mode == "global":
            return np.zeros((0, 3), np.float32)
        pts = np.asarray(points, np.float32).copy()
        if pts.size > 0:
            pts[:, 2] = 1.0
        return pts

    H0, W0 = img_bgr.shape[:2]
    img_t, (_, _), (H_in, W_in), _, _ = extractor._preprocess_manual(img_bgr)
    img_feat, img_pe, high_res = extractor._get_image_embed(img_t)

    pts_np = np.asarray(pts_np, np.float32) if pts_np is not None else np.zeros((0, 3), np.float32)
    if resume_mode == "head":
        pts_np = _prep_bg_points(pts_np)

    if pts_np.size == 0:
        mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
        coords, labels = None, None
    else:
        coords, labels = _map_points_to_sam_input(pts_np, H0, W0, H_in, W_in)
        coords = coords.to(img_feat.device); labels = labels.to(img_feat.device)
        if labels.max() <= 0:
            mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
        else:
            sp, dp = extractor._encode_prompts(coords, labels)
            mask_logits = extractor._decode_mask(img_feat, img_pe, sp, dp, high_res)
            mask = torch.sigmoid(mask_logits)
            if mask.shape[-2:] != img_feat.shape[-2:]:
                mask = F.interpolate(mask, size=img_feat.shape[-2:], mode="bilinear", align_corners=False)
            if (not torch.isfinite(mask).all()) or (mask.sum() <= 1e-5):
                mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)

    feat = (img_feat * mask).flatten(2).sum(dim=-1) / (mask.flatten(2).sum(dim=-1) + 1e-6)
    feat = feat.squeeze(0)

    # SAM输入尺寸下的可视化底图
    img_rgb = (img_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    vis_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # (H_in, W_in, 3)

    mask2d = mask.squeeze().detach().float().cpu().numpy()
    if mask2d.shape != (vis_bgr.shape[0], vis_bgr.shape[1]):
        mask2d = cv2.resize(mask2d, (vis_bgr.shape[1], vis_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 把原图点缩放到SAM输入尺寸(用于叠加/排查)
    scale, pad_x, pad_y, _, _ = _compute_resize_and_padding(H0, W0, H_in, W_in)
    pts_vis = []
    for p in np.asarray(pts_np, np.float32):
        x = float(np.clip(p[0] * scale + pad_x, 0, W_in - 1))
        y = float(np.clip(p[1] * scale + pad_y, 0, H_in - 1))
        pts_vis.append((x, y, int(p[2] > 0)))

    return vis_bgr, mask2d, feat, pts_vis


@torch.no_grad()
def visualize_samples(
    extractor,
    head: nn.Module,
    df: pd.DataFrame,
    label_map: Dict,
    out_dir: Path,
    num_vis: int = 50,
    conf_thr: float = 0.5,
    topk: int = 3,
    device: str = "cuda",
    # 兼容旧调用签名（不在函数内使用）
    resume_mode: str = "head",
    bg_mask_mode: str = "mix",
    bg_mix_p: float = 0.5,
    # 新参数
    apply_logit: Tuple[Optional[torch.Tensor], float] = (None, 0.0),
    stratified_vis: bool = True,
):
    """
    左：验证集原图(不做任何归一化/颜色变换/缩放)；
    右：对同一原图，把 SAM2(按点提示)得到的掩码区域设为黑色，其他保持原图；并叠加点与文字。
    """
    ensure_dir(out_dir)
    id2tool = {int(v): str(k) for k, v in label_map["tool_to_id"].items()}

    # 去重 +（可选）分层抽样
    def _prepare_vis_df(df_in: pd.DataFrame) -> pd.DataFrame:
        d = df_in
        if "image_path" in d.columns:
            d = d.drop_duplicates(subset=["image_path"]).copy()
        d = d.sample(frac=1.0, random_state=42).reset_index(drop=True)
        if not stratified_vis or "tool" not in d.columns:
            return d.head(min(num_vis, len(d)))
        k = d["tool"].nunique()
        per_cls = max(1, num_vis // k)
        parts = [g.head(min(per_cls, len(g))) for _, g in d.groupby("tool")]
        dd = pd.concat(parts).sample(frac=1.0, random_state=123).reset_index(drop=True)
        return dd.head(min(num_vis, len(dd)))

    df_vis = _prepare_vis_df(df)
    log_prior, tau = apply_logit

    pbar = tqdm(range(len(df_vis)), ncols=100, desc="[vis] writing", leave=True)
    for i in pbar:
        row = df_vis.iloc[i]
        img_path = row["image_path"]; tool_name = str(row["tool"])
        tool_id  = int(label_map["tool_to_id"][tool_name])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)      # 左图=原图
        if img is None:
            continue

        # 点（原始 JSON）
        try:
            pts = json.loads(row["points_json"]) if isinstance(row["points_json"], str) and row["points_json"].strip() else []
        except Exception:
            pts = []
        pts_np = np.asarray(pts, np.float32) if len(pts) else np.zeros((0, 3), np.float32)

        # 右图：SAM2 按点生成掩码 → 置黑 → 分类
        pred_id, prob_vec, right_img, _mask2d_orig = infer_and_classify_with_points(
            extractor, head, img, pts_np,
            device=device, conf_thr=conf_thr,
            log_prior=log_prior, tau=tau,
            draw_points=True,
        )

        # 写文字到右图
        top_idx = prob_vec.argsort()[::-1][:topk]
        y0 = 28
        cv2.putText(right_img, f"GT: {tool_name} (p={prob_vec[tool_id]:.3f})",
                    (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        y = y0 + 28
        for rank, j in enumerate(top_idx, 1):
            cls_name = id2tool.get(int(j), str(int(j)))
            cv2.putText(right_img, f"Top{rank}: {cls_name}  {prob_vec[j]:.3f}",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 24

        # 左右拼接（左=原图，右=黑遮罩）
        side_by_side = np.concatenate([img, right_img], axis=1)
        cv2.imwrite(str(out_dir / f"vis_{i:04d}.jpg"), side_by_side)

    print(f"[VIS] saved {len(df_vis)} images to: {out_dir.resolve()}")
    try:
        with open(out_dir / "index.txt", "w", encoding="utf-8") as f:
            for i in range(len(df_vis)):
                f.write(str((out_dir / f"vis_{i:04d}.jpg").resolve()) + "\n")
    except Exception as e:
        print(f"[WARN] write index.txt failed: {e}")

@torch.no_grad()
def visualize_points_only(
    df: pd.DataFrame,
    out_dir: Path,
    num_vis: int = 50,
    stratified_vis: bool = True,
):
    ensure_dir(out_dir)

    # 复用你现成的采样逻辑
    def _prepare_vis_df(df_in: pd.DataFrame) -> pd.DataFrame:
        d = df_in
        if "image_path" in d.columns:
            d = d.drop_duplicates(subset=["image_path"]).copy()
        d = d.sample(frac=1.0, random_state=42).reset_index(drop=True)
        if not stratified_vis or "tool" not in d.columns:
            return d.head(min(num_vis, len(d)))
        k = d["tool"].nunique()
        per_cls = max(1, num_vis // k)
        parts = [g.head(min(per_cls, len(g))) for _, g in d.groupby("tool")]
        dd = pd.concat(parts).sample(frac=1.0, random_state=123).reset_index(drop=True)
        return dd.head(min(num_vis, len(dd)))

    df_vis = _prepare_vis_df(df)

    pbar = tqdm(range(len(df_vis)), ncols=100, desc="[vis] points-only", leave=True)
    for i in pbar:
        row = df_vis.iloc[i]
        img_path = row["image_path"]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # 解析点
        try:
            pts = json.loads(row["points_json"]) if isinstance(row["points_json"], str) and row["points_json"].strip() else []
        except Exception:
            pts = []
        pts_np = np.asarray(pts, np.float32) if len(pts) else np.zeros((0, 3), np.float32)

        # 左：原图；右：原图 + 叠加点（绿=正，红=负）
        left  = img
        right = img.copy()
        H, W = right.shape[:2]

        for p in pts_np:
            x = int(np.clip(round(float(p[0])), 0, W-1))
            y = int(np.clip(round(float(p[1])), 0, H-1))
            is_pos = int(float(p[2]) > 0)
            c = (0, 255, 0) if is_pos else (0, 0, 255)  # green / red
            cv2.circle(right, (x, y), 5, c, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(right, (x, y), 6, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        side_by_side = np.concatenate([left, right], axis=1)
        cv2.imwrite(str(out_dir / f"vis_pts_{i:04d}.jpg"), side_by_side)

    print(f"[VIS][points-only] saved {len(df_vis)} images to: {out_dir.resolve()}")
    try:
        with open(out_dir / "index_points_only.txt", "w", encoding="utf-8") as f:
            for i in range(len(df_vis)):
                f.write(str((out_dir / f"vis_pts_{i:04d}.jpg").resolve()) + "\n")
    except Exception as e:
        print(f"[WARN] write index_points_only.txt failed: {e}")





# ========================= Main =========================

def _choose_split_csv() -> Path:
    test_csv = SMALLFILE_ROOT / "test_manifest.csv"
    val_csv  = SMALLFILE_ROOT / "val_manifest.csv"
    mf_csv   = SMALLFILE_ROOT / "manifest.csv"
    if test_csv.exists(): return test_csv
    if val_csv.exists():  return val_csv
    return mf_csv

def main():
    ap = argparse.ArgumentParser()
    # which dataset to eval
    ap.add_argument("--eval-set", choices=["prepared", "ood_raw"], default="prepared",
                    help="'prepared' uses existing CSV; 'ood_raw' builds OOD from prompts.json")
    ap.add_argument("--videos", type=str,
                    default="video01,video02,video03,video04,video05,video06,video07,video08,video09,video10",
                    help="Only for --eval-set ood_raw; comma-separated video ids.")
    ap.add_argument("--raw-root", type=str,
                    default="/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data",
                    help="Root of annotated_data for OOD.")
    ap.add_argument("--ood-out-csv", type=str, default=str(SMALLFILE_ROOT / "ood_manifest.csv"))
    ap.add_argument("--ood-label-map", type=str, default=str(SMALLFILE_ROOT / "label_map_ood.json"))

    # model & ckpt
    ap.add_argument("--resume-mode", choices=["head", "full"], default="head",
                    help="'head': frozen SAM2 + head-only ckpt (best_head.pt). 'full': finetuned SAM2+head (best_full_finetune.pt).")
    ap.add_argument("--ckpt", type=str, default=str(CKPT_ROOT / "best_head.pt"),
                    help="Path to checkpoint. head-mode: best_head.pt; full-mode: best_full_finetune.pt")
    ap.add_argument("--head-override", type=str, default=None,
                    help="Optional override ('linear','mlp','mlp_bn','cosine') if ckpt args missing.")

    # sam2 backbones
    ap.add_argument("--sam2-cfg", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_l.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_large.pt"))

    # general
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-vis", type=int, default=50)
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out-dir", type=str, default=str(CKPT_ROOT))

    # test-time options for alignment
    ap.add_argument("--bg-mask-mode", choices=["pos","global","mix"], default="mix",
                    help="Test-time background mask policy to match training.")
    ap.add_argument("--bg-mix-p", type=float, default=0.5)
    ap.add_argument("--apply-ckpt-logit-adjust", action="store_true",
                    help="Apply logit-adjust in eval/vis if ckpt used it during training.")
    ap.add_argument("--stratified-vis", action="store_true",
                    help="Make the visualization set roughly class-balanced and deduplicated.")
    ap.add_argument("--vis-points-only", action="store_true",
                help="Only visualize prompts (green=positive, red=negative), skip SAM2 inference & classification.")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- build/select dataset ---------
    if args.eval_set == "prepared":
        label_map_path = SMALLFILE_ROOT / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"[prepared] missing label_map: {label_map_path}")
        label_map = load_label_map(label_map_path)
        split_csv = _choose_split_csv()
        if not split_csv.exists():
            raise FileNotFoundError(f"[prepared] cannot find CSV: {split_csv}")
        df = pd.read_csv(split_csv)
        print(f"[INFO] Evaluating (prepared): {split_csv}  (#rows={len(df)})")
        ds = FramePointDataset(split_csv, label_map_path, resize=None)
    else:
        # OOD raw
        ood_label_map_path = Path(args.ood_label_map)
        if not ood_label_map_path.exists():
            _save_json({"tool_to_id": TOOL_TO_ID_OOD}, ood_label_map_path)
            print(f"[SAVE] OOD label_map -> {ood_label_map_path}")
        else:
            lm_tmp = _load_json(ood_label_map_path)
            assert lm_tmp.get("tool_to_id", {}) == TOOL_TO_ID_OOD, "OOD label_map mismatch"
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]
        ood_csv = Path(args.ood_out_csv)
        if not ood_csv.exists():
            _build_ood_manifest(Path(args.raw_root), videos, ood_csv)
        else:
            print(f"[INFO] Using existing OOD manifest: {ood_csv}")
        df = pd.read_csv(ood_csv)
        print(f"[INFO] Evaluating (OOD raw): {ood_csv}  (#rows={len(df)})")
        label_map_path = ood_label_map_path
        label_map = _load_json(label_map_path)
        ds = FramePointDataset(ood_csv, label_map_path, resize=None)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_varlen, pin_memory=True)

        # --------- points-only visualization (no SAM2, no head) ---------
    if args.vis_points_only:
        vis_dir = Path(args.out_dir) / ("vis_points_only_ood" if args.eval_set=="ood_raw" else "vis_points_only")
        visualize_points_only(
            pd.read_csv(_choose_split_csv()) if args.eval_set=="prepared" else df,
            vis_dir,
            num_vis=args.num_vis,
            stratified_vis=args.stratified_vis,
        )
        return

    # --------- build extractor ---------
    if args.resume_mode == "head":
        if Sam2OfficialWrapper is None:
            raise RuntimeError("Sam2OfficialWrapper not available. Check frozen training script import.")
        extractor = Sam2OfficialWrapper(args.sam2_cfg, args.sam2_ckpt, device=device, cache_size=128)
        extractor.eval()
    else:
        if FineTuneSam2Wrapper is None:
            raise RuntimeError("FineTuneSam2Wrapper not available. Check finetune training script import.")
        extractor = FineTuneSam2Wrapper(args.sam2_cfg, args.sam2_ckpt, device=device)
        extractor.set_trainable("none")  # eval mode; don't update
        extractor.eval()

    # --------- probe feature dim ---------
    probe = next(iter(dl))
    with torch.no_grad():
        feat_probe = extractor(probe["images"][:1], probe["points"][:1], probe.get("meta"))
    in_dim = int(feat_probe.shape[-1])
    n_classes = len(label_map["tool_to_id"])

    # --------- load ckpt + head ---------
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ck_args = ckpt.get("args", {})
    in_dim_ck = int(ckpt.get("in_dim", in_dim))
    n_classes_ck = int(ckpt.get("n_classes", n_classes))
    if n_classes_ck != n_classes:
        print(f"[WARN] n_classes mismatch: ckpt={n_classes_ck}, current={n_classes}. Using current={n_classes}.")
        n_classes_ck = n_classes

    head = _build_head_from_ckpt_args(in_dim_ck, n_classes_ck, ck_args, override_head=args.head_override).to(device)

    if args.resume_mode == "full":
        # FULL: expect {sam2_state, head_state}
        sam2_state = ckpt.get("sam2_state", None)
        head_state = ckpt.get("head_state", None) or ckpt
        if sam2_state is not None:
            try:
                extractor.load_state_dict(sam2_state, strict=False)
            except Exception:
                # For our wrapper, parameters sit under extractor.model.*
                extractor.load_state_dict(sam2_state, strict=False)
            print(f"[RESUME] Loaded finetuned SAM2 from: {ckpt_path}")
        else:
            print(f"[WARN] No 'sam2_state' in {ckpt_path}; proceeding with base SAM2 weights.")
        head.load_state_dict(head_state, strict=False)
        print(f"[RESUME] Loaded head state (full mode) from: {ckpt_path}")
    else:
        # HEAD-ONLY
        state = ckpt.get("head_state", ckpt)
        head.load_state_dict(state, strict=False)
        print(f"[RESUME] Loaded head state (head-only mode) from: {ckpt_path}")

    # --------- logit-adjust prior (optional) ---------
    tau = 0.0
    log_prior = None
    if args.apply_ckpt_logit_adjust:
        tau = float(ck_args.get("logit-adjust", 0.0))
        train_csv_for_prior = SMALLFILE_ROOT / "train_manifest.csv"
        if (tau > 0.0) and train_csv_for_prior.exists():
            tool2id = label_map["tool_to_id"]
            df_train = pd.read_csv(train_csv_for_prior)
            ids = [tool2id[t] for t in df_train["tool"] if t in tool2id]
            max_id = max(tool2id.values())
            cnt = np.bincount(ids, minlength=max_id+1).astype(np.float64)
            pri = cnt / max(1.0, cnt.sum())
            log_prior = torch.log(torch.tensor(pri + 1e-12, device=device, dtype=torch.float32))
            print(f"[LOGIT-ADJUST] tau={tau:.3f}  priors={pri.round(4)}")
        else:
            print("[LOGIT-ADJUST] skipped (tau<=0 or no train_manifest.csv).")

    # --------- evaluate ---------
    test_loss, overall_acc, per_class_acc, per_class_cnt, y_true, y_pred = evaluate_batchwise(
        extractor, head, dl, device,
        log_prior=log_prior, tau=tau,
        bg_mask_mode=args.bg_mask_mode, bg_mix_p=args.bg_mix_p,
        resume_mode=args.resume_mode, topk_eval=1
    )

    id2tool = {int(v): str(k) for k, v in label_map["tool_to_id"].items()}
    print(f"\n=== Overall ===\nLoss: {test_loss:.4f}  Acc: {overall_acc:.4f}  (#samples={len(y_true)})")
    print("\n=== Per-class Acc ===")
    for cid in sorted(per_class_acc.keys()):
        cname = id2tool.get(cid, str(cid))
        cnt   = per_class_cnt.get(cid, 0)
        print(f"{cid:3d} {cname:>20s}: acc={per_class_acc[cid]:.4f}  (n={cnt})")

    # --------- visualize ---------
    vis_dir = Path(args.out_dir) / ("vis_test_ood" if args.eval_set=="ood_raw" else "vis_test")
    visualize_samples(extractor, head, pd.read_csv(_choose_split_csv()) if args.eval_set=="prepared" else df,
                      label_map, vis_dir,
                      num_vis=args.num_vis, conf_thr=args.conf_thr, topk=args.topk, device=device,
                      resume_mode=args.resume_mode, bg_mask_mode=args.bg_mask_mode, bg_mix_p=args.bg_mix_p,
                      apply_logit=(log_prior, tau), stratified_vis=args.stratified_vis)

if __name__ == "__main__":
    main()

# 只训练classifier
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --eval-set prepared \
#   --resume-mode head \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/best_head.pt \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --batch-size 128 \
#   --bg-mask-mode mix --bg-mix-p 0.5 \
#   --apply-ckpt-logit-adjust \
#   --stratified-vis \
#   --num-vis 50

#只可视化点
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --eval-set prepared \
#   --resume-mode head \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/best_head.pt \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --num-vis 50 \
#   --vis-points-only


#finetune整个sam2
#   python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --eval-set prepared \
#   --resume-mode full \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/best_full_finetune.pt \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --batch-size 128 \
#   --bg-mask-mode mix --bg-mix-p 0.5 \
#   --apply-ckpt-logit-adjust \
#   --stratified-vis \
#   --num-vis 50

# 只训练classifier + finetune prompt、mask encoder
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --eval-set prepared \
#   --resume-mode full \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/pmstage_best.pt \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --bg-mask-mode mix --bg-mix-p 0.5 \
#   --apply-ckpt-logit-adjust \
#   --batch-size 128 --workers 4 \
#   --num-vis 50 --conf-thr 0.5 --topk 3 \
#   --out-dir /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier



