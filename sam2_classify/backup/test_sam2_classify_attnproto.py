#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test AttnProto (SAM2 frozen) classifier.

Loads:
- Frozen SAM2 wrapper from your training script: train_sam2_classify.Sam2OfficialWrapper
- AttnProtoPooler (this file) + CosineClassifier (this file)
- CKPT fields expected: {"pooler_state","head_state","in_dim","n_classes","tool_to_id","args"}

Eval:
- prepared CSVs under /home/wcheng31/sam2_classify/config (train/val/test or manifest.csv)
- per-batch feature extraction: image_feat + mask -> adaptive pooling to (Hg,Wg)
- AttnProtoPooler -> (C,) -> Cos classifier -> logits
- accuracy + CE loss; (optional) logit-adjust at test time

"""

import os, json, argparse, random, math, hashlib
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

# ---------------- Paths ----------------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"

# --------------- Utils -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_label_map(p: Path) -> Dict[str, Dict[str, int]]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------- Dataset ----------------
class FramePointDataset(Dataset):
    """Eval dataset: keep rows with non-empty points_json."""
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
            "targets": tool_id,
            "meta": {"image_path": row["image_path"], "tool": tool, "clip_name": row.get("clip_name","")}
        }

def collate_varlen(batch):
    images  = [b["image"]  for b in batch]
    points  = [b["points"] for b in batch]
    targets = torch.tensor([b["targets"] for b in batch], dtype=torch.long)
    metas   = [b["meta"]   for b in batch]
    return {"images": images, "points": points, "targets": targets, "meta": metas}

# ---------- Import frozen SAM2 wrapper ----------
import sys
sys.path.append(str(SMALLFILE_ROOT))
try:
    from train_sam2_classify import Sam2OfficialWrapper  # 你冻结训练脚本里定义的 wrapper
except Exception as e:
    raise RuntimeError(f"Cannot import Sam2OfficialWrapper from train_sam2_classify.py under {SMALLFILE_ROOT}: {e}")

# ---------- AttnProto Pooler & Cosine Head ----------
class AttnProtoPooler(nn.Module):
    """
    多查询注意力原型池化：
      - 输入: feats (B,C,Hg,Wg), mask (B,1,Hg,Wg) ∈ [0,1]
      - 过程: token = (Hg*Wg, C)，用 K 个查询向量与 token 点积 -> softmax 注意力（masked）
      - 输出: 将 K 个原型平均或 concat（此处平均）-> (B,C)
    """
    def __init__(self, in_dim: int, num_queries: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.num_queries = int(num_queries)
        self.query = nn.Parameter(torch.randn(self.num_queries, in_dim))
        nn.init.normal_(self.query, mean=0.0, std=1.0 / math.sqrt(in_dim))

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # feat: (B,C,Hg,Wg) ; mask: (B,1,Hg,Wg)
        B, C, Hg, Wg = feat.shape
        x = feat.view(B, C, Hg*Wg).transpose(1, 2)        # (B, N, C)
        m = mask.view(B, 1, Hg*Wg)                        # (B,1,N)
        m = m.clamp_min(0).clamp_max(1)

        # 归一化 token，查询
        x_n = F.normalize(x, dim=-1)                      # (B,N,C)
        q_n = F.normalize(self.query, dim=-1)             # (K,C)
        att = torch.einsum('bnc,kc->bnk', x_n, q_n)       # (B,N,K)

        # masked softmax over N
        very_neg = torch.finfo(att.dtype).min / 2
        att = att.transpose(1, 2)                         # (B,K,N)
        att = att.masked_fill(m <= 0.0, very_neg)
        att = F.softmax(att, dim=-1)                      # (B,K,N)

        # 聚合到原型: (B,K,C)
        proto = torch.einsum('bkn,bnc->bkc', att, x)
        # 平均 K 个原型
        out = proto.mean(dim=1)                           # (B,C)
        return out

class CosineClassifier(nn.Module):
    """测试时 margin=0 的余弦分类器；训练时的 CosFace 权重可直接 load."""
    def __init__(self, in_dim: int, n_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x_n, w_n)

# ---------- Background policy helper ----------
def _prep_points_for_bg(points: np.ndarray, tool_id: int, bg_mask_mode: str, bg_mix_p: float) -> np.ndarray:
    if tool_id != 0:
        return points
    mode = bg_mask_mode
    if mode == "mix":
        mode = "global" if (random.random() < float(bg_mix_p)) else "pos"
    if mode == "global":
        return np.zeros((0,3), np.float32)
    pts = np.asarray(points, np.float32).copy()
    if pts.size > 0:
        pts[:, 2] = 1.0
    return pts

# ---------- AttnProto Extractor Adapter ----------
class AttnProtoExtractor(nn.Module):
    """
    将你冻结的 SAM2 wrapper 适配成：forward(images, points, metas) -> (B,C) 特征
    步骤：
      - 用 wrapper 的内部方法得到 img_feat(C,H',W') 与 mask(1,H',W')
      - 自适应到固定网格 (Hg,Wg)（同训练）
      - 喂给 AttnProtoPooler -> (B,C)
    """
    def __init__(self, sam2_wrapper: Sam2OfficialWrapper, pooler: AttnProtoPooler,
                 Hg: int = 14, Wg: int = 14, bg_mask_mode: str = "mix", bg_mix_p: float = 0.5):
        super().__init__()
        self.sam2 = sam2_wrapper
        self.pooler = pooler
        self.Hg, self.Wg = int(Hg), int(Wg)
        self.bg_mask_mode = bg_mask_mode
        self.bg_mix_p = bg_mix_p

    @torch.no_grad()
    def _encode_one(self, img_bgr: np.ndarray, pts_np: np.ndarray):
        # 复用 wrapper 的内部预处理 + 编码 + mask 逻辑
        img_t, _, _, sy, sx = self.sam2._preprocess_manual(img_bgr)
        img_feat, img_pe, high_res = self.sam2._get_image_embed(img_t)

        if pts_np is None or len(pts_np) == 0:
            mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
        else:
            coords = self.sam2._map_points_scale_xy(pts_np, sy, sx).to(img_feat.device)
            labels = torch.from_numpy((pts_np[:,2] > 0).astype(np.int64)).unsqueeze(0).to(img_feat.device)
            if labels.max() <= 0:
                mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
            else:
                sp, dp = self.sam2._encode_prompts(coords, labels)
                mask_logits = self.sam2._decode_mask(img_feat, img_pe, sp, dp, high_res)
                mask = torch.sigmoid(mask_logits)
                if mask.shape[-2:] != img_feat.shape[-2:]:
                    mask = F.interpolate(mask, size=img_feat.shape[-2:], mode="bilinear", align_corners=False)
                if (not torch.isfinite(mask).all()) or (mask.sum() <= 1e-5):
                    mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
        return img_feat, mask

    @torch.no_grad()
    def forward(self, images_bgr: List[np.ndarray], points_list: List[np.ndarray], metas=None) -> torch.Tensor:
        feats_all = []
        for i in range(len(images_bgr)):
            img = images_bgr[i]
            pts = points_list[i]
            # 背景策略与训练对齐（仅在 tool_id==0 时生效）
            tool_id = None
            if metas is not None and i < len(metas):
                # 如果上层没传 target，这里不强依赖，默认用 points 原样
                pass
            # 无法直接拿 tool_id，只能在上层统一处理；这里先不过滤

            img_feat, mask = self._encode_one(img, pts)
            # 自适应到 (Hg, Wg)
            img_feat_g = F.adaptive_avg_pool2d(img_feat, output_size=(self.Hg, self.Wg))  # (1,C,Hg,Wg)
            mask_g     = F.adaptive_avg_pool2d(mask,     output_size=(self.Hg, self.Wg))  # (1,1,Hg,Wg)
            # 归一化 mask，让数值更稳
            mask_sum = mask_g.flatten(2).sum(-1).clamp_min(1e-6)  # (1,1)
            mask_g = mask_g / mask_sum.view(1,1,1,1)

            feat_i = self.pooler(img_feat_g, mask_g)  # (1,C) -> squeeze
            feats_all.append(feat_i.squeeze(0))
            # 释放
            del img_feat, mask, img_feat_g, mask_g
        return torch.stack(feats_all, dim=0)

# ---------- Eval (batchwise, 复用你之前 tester 的思路) ----------
@torch.no_grad()
def evaluate_batchwise(extractor, head: nn.Module, loader: DataLoader, device: str,
                       log_prior: Optional[torch.Tensor], tau: float) -> Tuple[float, float, Dict[int,float], Dict[int,int], np.ndarray, np.ndarray]:
    head.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_n, total_correct = 0.0, 0, 0
    y_true, y_pred = [], []

    pbar = tqdm(loader, total=len(loader), ncols=100, desc="[test] eval", leave=True)
    for batch in pbar:
        imgs, pts_list, y = batch["images"], batch["points"], batch["targets"].to(device)

        feats = extractor(imgs, pts_list, batch.get("meta"))
        logits = head(feats)

        if (log_prior is not None) and (tau is not None) and (tau > 0):
            logits = logits - float(tau) * log_prior.view(1, -1).to(logits.device)

        loss = ce(logits, y)
        pred = logits.argmax(dim=1)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        total_correct += (pred == y).sum().item()

        y_true.append(y.detach().cpu().numpy()); y_pred.append(pred.detach().cpu().numpy())
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

# ---------- CSV pick ----------
def _pick_eval_csv() -> Path:
    test_csv = SMALLFILE_ROOT / "test_manifest.csv"
    val_csv  = SMALLFILE_ROOT / "val_manifest.csv"
    mf_csv   = SMALLFILE_ROOT / "manifest.csv"
    if test_csv.exists(): return test_csv
    if val_csv.exists():  return val_csv
    return mf_csv

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to attnproto ckpt, e.g., best_attnproto_head.pt")
    ap.add_argument("--sam2-cfg", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_l.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_large.pt"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grid", type=str, default="14,14", help="adaptive pooling grid Hg,Wg")
    ap.add_argument("--bg-mask-mode", choices=["pos","global","mix"], default="mix")
    ap.add_argument("--bg-mix-p", type=float, default=0.5)
    ap.add_argument("--apply-ckpt-logit-adjust", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(123)

    # dataset
    label_map_path = SMALLFILE_ROOT / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label_map: {label_map_path}")
    split_csv = _pick_eval_csv()
    if not split_csv.exists():
        raise FileNotFoundError(f"Cannot find eval CSV: {split_csv}")
    ds = FramePointDataset(split_csv, label_map_path, resize=None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_varlen, pin_memory=True)

    label_map = load_label_map(label_map_path)
    n_classes = len(label_map["tool_to_id"])
    id2tool   = {int(v): k for k, v in label_map["tool_to_id"].items()}

    # frozen SAM2
    sam2 = Sam2OfficialWrapper(args.sam2_cfg, args.sam2_ckpt, device=device, cache_size=128)
    sam2.eval()

    # probe C dim
    probe = next(iter(dl))
    with torch.no_grad():
        img_t, _, _, sy, sx = sam2._preprocess_manual(probe["images"][0])
        img_feat, img_pe, high_res = sam2._get_image_embed(img_t)
    C = int(img_feat.shape[1]); del img_t, img_feat, img_pe, high_res

    # build pooler + head
    Hg, Wg = [int(x.strip()) for x in args.grid.split(",")]
    pooler = AttnProtoPooler(in_dim=C, num_queries=4).to(device)
    head   = CosineClassifier(in_dim=C, n_classes=n_classes, scale=16.0).to(device)

    # load ckpt
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ck_args = ckpt.get("args", {})

    # override hyper from ckpt if present
    if "scale" in ck_args:
        head.scale = float(ck_args["scale"])
    if "num_queries" in ck_args:
        # 需要与训练时一致；如不一致则重新构造再 load
        nq = int(ck_args["num_queries"])
        if nq != pooler.num_queries:
            pooler = AttnProtoPooler(in_dim=C, num_queries=nq).to(device)

    pooler_state = ckpt.get("pooler_state", None)
    head_state   = ckpt.get("head_state", None)
    if pooler_state is None or head_state is None:
        raise RuntimeError("Invalid ckpt: expect keys 'pooler_state' and 'head_state'.")

    pooler.load_state_dict(pooler_state, strict=True)
    head.load_state_dict(head_state, strict=False)
    print(f"[RESUME] loaded pooler+head from {ckpt_path}")

    # logit-adjust (optional, if used in training)
    tau = 0.0
    log_prior = None
    if args.apply_ckpt_logit_adjust:
        tau = float(ck_args.get("logit-adjust", 0.0))
        train_csv_for_prior = SMALLFILE_ROOT / "train_manifest.csv"
        if (tau > 0.0) and train_csv_for_prior.exists():
            tool2id = label_map["tool_to_id"]
            df_train = pd.read_csv(train_csv_for_prior)
            ids = [tool2id[t] for t in df_train["tool"] if t in tool2id]
            cnt = np.bincount(ids, minlength=max(tool2id.values())+1).astype(np.float64)
            pri = cnt / max(1.0, cnt.sum())
            log_prior = torch.log(torch.tensor(pri + 1e-12, device=device, dtype=torch.float32))
            print(f"[LOGIT-ADJUST] tau={tau:.3f} priors={np.round(pri,4)}")
        else:
            print("[LOGIT-ADJUST] skipped (tau<=0 or no train_manifest.csv).")

    # build extractor adapter
    extractor = AttnProtoExtractor(sam2, pooler, Hg=Hg, Wg=Wg,
                                   bg_mask_mode=args.bg_mask_mode, bg_mix_p=args.bg_mix_p)

    # EVAL
    test_loss, overall_acc, per_class_acc, per_class_cnt, y_true, y_pred = evaluate_batchwise(
        extractor, head, dl, device, log_prior, tau
    )

    print(f"\n=== Overall ===\nLoss: {test_loss:.4f}  Acc: {overall_acc:.4f}  (#samples={len(y_true)})")
    print("\n=== Per-class Acc ===")
    for cid in sorted(per_class_acc.keys()):
        cname = id2tool.get(cid, str(cid))
        cnt   = per_class_cnt.get(cid, 0)
        print(f"{cid:3d} {cname:>20s}: acc={per_class_acc[cid]:.4f}  (n={cnt})")

if __name__ == "__main__":
    main()


# python /home/wcheng31/sam2_classify/test_sam2_classify_attnproto.py \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/best_attnproto_head.pt \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --batch-size 128 --workers 4 \
#   --grid 14,14 \
#   --bg-mask-mode mix --bg-mix-p 0.5 \
#   --apply-ckpt-logit-adjust
