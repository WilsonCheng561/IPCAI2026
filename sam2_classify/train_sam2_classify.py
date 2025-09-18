#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增加类别均衡的训练方式, Sam2 frozen
Train a small classification head on top of *frozen* SAM2 features with point prompts.

Data root (default): /mnt/disk1/haoding/sam2_data
Expected files:
  - /mnt/disk1/haoding/sam2_data/manifest.csv
  - /mnt/disk1/haoding/sam2_data/label_map.json

Each manifest row (from your data prep) corresponds to ONE (frame x tool) sample:
  image_path, tool, points_json=[[x,y,label], ...], ...  (label: 1=positive tip/anchor, 0=negative/contact or ignored)

What this script does:
  1) Deterministic split with reproducible seed. If train/val/test CSVs already exist, use them.
     Otherwise, we **split by clip_name** (non-overlapping) to avoid leakage across splits.
  2) Freeze SAM2; for each image:
        - encode image -> image_embed (C,H',W') with proper pixel mean/std normalization
        - encode point prompts -> mask logits -> sigmoid -> (1,H',W')
        - masked average pooling over image_embed -> (C,)
  3) Train a small MLP/linear/cosine head for tool classification (only head has gradients).
  4) Early stopping on val acc; save best head to out_dir (default: PRETRAIN_ROOT/cwz/sam2_classifier/best_head.pt)
  5) Also dump deterministic train/val/test CSVs if they don't exist.

Notes:
- For geometric consistency, prefer --resize=None (default). The SAM2 wrapper handles model-legal sizing.
- We use a small **LRU cache** inside the SAM2 wrapper keyed by image_path (or a short image signature)
  to avoid re-encoding identical frames within and across batches.

Author: Wenzheng Cheng
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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):  # graceful fallback
        return x


# ----------------------------- Global path overrides -----------------------------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify")  # csv、日志等
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"
DATASET_ROOT   = Path("/projects/surgical-video-digital-twin/datasets/sam2_classifier")


# ----------------------------- Utils -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_label_map(p: Path) -> Dict[str, Dict[str, int]]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# === Split by clip_name (non-overlap) to avoid leakage ===
def _stable_train_val_test_split(df: pd.DataFrame, seed: int = 42,
                                 train_ratio: float = 0.8, val_ratio: float = 0.1):
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for clip, g in df.groupby("clip_name"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val   = min(n_val, max(0, n - n_train))
        tr = idx[:n_train]
        va = idx[n_train:n_train+n_val]
        te = idx[n_train+n_val:]
        train_parts.append(g.loc[tr])
        val_parts.append(g.loc[va])
        test_parts.append(g.loc[te])
    df_train = pd.concat(train_parts).reset_index(drop=True) if train_parts else df.iloc[[]].copy()
    df_val   = pd.concat(val_parts).reset_index(drop=True)   if val_parts   else df.iloc[[]].copy()
    df_test  = pd.concat(test_parts).reset_index(drop=True)  if test_parts  else df.iloc[[]].copy()
    return df_train, df_val, df_test

class TrainingLogger:
    def __init__(self, filepath: Path):
        ensure_dir(filepath.parent)
        self.f = open(str(filepath), "a", buffering=1, encoding="utf-8")
    def write(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.f.write(f"[{ts}] {msg}\n")
    def close(self):
        try: self.f.close()
        except Exception: pass


# ---- Class stats / balancing helpers ----
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
    """CE 权重：1/log(1+n)，再做均值标准化到 ~1 的尺度"""
    w = 1.0 / np.log(1.0 + np.maximum(counts, 1.0))
    w = w * (len(w) / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def _sampling_weights_from_counts(counts: np.ndarray, alpha: float = 0.5, bg_factor: float = 0.3) -> np.ndarray:
    """
    采样权重：w_c = (n_c)^(-alpha)，并对 background(0类) 乘以 bg_factor (<1 抑制)
    不做归一化，WeightedRandomSampler 只看相对大小。
    """
    eps = 1e-12
    w = (np.maximum(counts, eps)) ** (-float(alpha))
    if len(w) > 0:
        w[0] *= float(bg_factor)   # 假定 0 类是 background
    return w


# ----------------------------- Dataset -----------------------------

class FramePointDataset(Dataset):
    def __init__(self, manifest_csv: Path, label_map_json: Path,
                 resize: Optional[int] = None, bg_mask_mode: str = "mix"):
        super().__init__()
        self.df = pd.read_csv(manifest_csv)
        self.label_map = load_label_map(label_map_json)
        self.tool2id = self.label_map["tool_to_id"]
        self.resize = resize
        self.bg_mask_mode = bg_mask_mode  # <<< 新增

        def has_points(s: str) -> bool:
            try:
                arr = json.loads(s) if isinstance(s, str) and s.strip() else []
                return len(arr) > 0
            except Exception:
                return False
        self.df = self.df[self.df["points_json"].apply(has_points)].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def _load_img(self, p: str) -> np.ndarray:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(p)
        if self.resize and self.resize > 0:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        return img  # BGR uint8

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_img(row["image_path"])
        H, W = img.shape[:2]
        pts = json.loads(row["points_json"]) if isinstance(row["points_json"], str) and row["points_json"].strip() else []
        pts_out = []
        for p in pts:
            if len(p) < 2: continue
            x = float(np.clip(p[0], 0, W-1)); y = float(np.clip(p[1], 0, H-1))
            label = 1.0 if (len(p) >= 3 and float(p[2]) > 0) else 0.0
            pts_out.append([x, y, label])

        tool = str(row["tool"])
        tool_id = self.tool2id.get(tool, None)
        if tool_id is None:
            for k, v in self.tool2id.items():
                if str(k) == tool:
                    tool_id = v; break
        if tool_id is None:
            raise KeyError(f"Tool '{tool}' not in label_map.json")

        # -------- BG 专用逻辑（关键修改）--------
        if int(tool_id) == 0:  # background
            mode = self.bg_mask_mode
            if mode == "pos":
                # 把 BG 的点全部改为正点（让 SAM 掩码吸附到这些点）
                for p in pts_out:
                    p[2] = 1.0
            elif mode == "global":
                # 不喂点，后续 extractor 里自动走“全图 GAP”
                pts_out = []
            else:  # mix：一半全图，一半正点
                if random.random() < 0.5:
                    pts_out = []
                else:
                    for p in pts_out:
                        p[2] = 1.0
        # ------------------------------------

        return {
            "image": img,
            "points": np.array(pts_out, dtype=np.float32) if pts_out else np.zeros((0,3), np.float32),
            "tool_id": int(tool_id),
            "meta": {
                "image_path": row["image_path"],
                "tool": tool,
                "frame_abs_index": int(row.get("frame_abs_index", -1)),
                "frame_idx_in_clip": int(row.get("frame_idx_in_clip", -1)),
                "clip_name": row.get("clip_name", ""),
                "task": row.get("task", "")
            }
        }


def collate_varlen(batch):
    images  = [b["image"]  for b in batch]
    points  = [b["points"] for b in batch]
    targets = torch.tensor([b["tool_id"] for b in batch], dtype=torch.long)
    metas   = [b["meta"]   for b in batch]
    return {"images": images, "points": points, "targets": targets, "meta": metas}


# ----------------------------- SAM2 Wrapper -----------------------------

class Sam2OfficialWrapper(nn.Module):
    def __init__(self, cfg: str, ckpt: str, device: str = "cuda", cache_size: int = 128):
        super().__init__()
        self.device = device
        from sam2.build_sam import build_sam2  # 按你的要求不做导入鲁棒性增强
        self.model = build_sam2(cfg, ckpt, device=device)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.verbose = False
        self._printed_resize = False
        self._norm_cached: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.cache: "OrderedDict[str, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], float, float]]" = OrderedDict()
        self.cache_size = int(cache_size)

    # --- geometry helpers ---
    def _infer_patch_conv(self):
        enc = getattr(self.model, "image_encoder", None)
        trunk = getattr(enc, "trunk", None)
        if trunk is None or not hasattr(trunk, "patch_embed") or not hasattr(trunk.patch_embed, "proj"):
            return 7, 7, 4, 4, 3, 3
        conv = trunk.patch_embed.proj
        k = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        s = conv.stride      if isinstance(conv.stride, tuple)      else (conv.stride, conv.stride)
        p = conv.padding     if isinstance(conv.padding, tuple)     else (conv.padding, conv.padding)
        return int(k[0]), int(k[1]), int(s[0]), int(s[1]), int(p[0]), int(p[1])

    def _infer_window_size(self) -> Optional[int]:
        enc = getattr(self.model, "image_encoder", None)
        trunk = getattr(enc, "trunk", None)
        for name in ["window_size", "win_size", "ws"]:
            if hasattr(trunk, name):
                v = getattr(trunk, name)
                if isinstance(v, int): return v
                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], int):
                    return int(v[0])
        return None

    def _tokens_for(self, n_pix: int, k: int, s: int, p: int) -> int:
        return int((n_pix + 2*p - k) // s + 1)
    def _size_for_tokens(self, n_tok: int, k: int, s: int, p: int) -> int:
        return int(s * (n_tok - 1) + k - 2 * p)

    def _legal_hw_from_orig(self, H0: int, W0: int):
        k_h,k_w,s_h,s_w,p_h,p_w = self._infer_patch_conv()
        win = self._infer_window_size() or 16
        t_h0 = max(1, self._tokens_for(H0, k_h, s_h, p_h))
        t_w0 = max(1, self._tokens_for(W0, k_w, s_w, p_w))
        t_h = max(win, (t_h0 // win) * win)
        t_w = max(win, (t_w0 // win) * win)
        H_in = self._size_for_tokens(t_h, k_h, s_h, p_h)
        W_in = self._size_for_tokens(t_w, k_w, s_w, p_w)
        sy = H_in / max(1, H0); sx = W_in / max(1, W0)
        if self.verbose and not self._printed_resize:
            print(f"[SAM2] orig=({H0},{W0}) -> in=({H_in},{W_in}), scale=({sy:.3f},{sx:.3f})")
            self._printed_resize = True
        return H_in, W_in, sy, sx

    # --- normalization ---
    def _get_norm(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._norm_cached is not None:
            return self._norm_cached
        for obj in [self.model, getattr(self.model, "image_encoder", None)]:
            if obj is None: continue
            pm = getattr(obj, "pixel_mean", None)
            ps = getattr(obj, "pixel_std",  None)
            if pm is not None and ps is not None:
                pm = torch.as_tensor(pm, dtype=torch.float32).view(1,3,1,1)
                ps = torch.as_tensor(ps, dtype=torch.float32).view(1,3,1,1)
                if pm.max() > 1.5 or ps.max() > 1.5:
                    pm = pm / 255.0
                    ps = ps / 255.0
                self._norm_cached = (pm.to(self.device), ps.to(self.device))
                return self._norm_cached
        pm = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        ps = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        self._norm_cached = (pm, ps)
        return self._norm_cached

    @torch.no_grad()
    def _preprocess_manual(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]
        H_in, W_in, sy, sx = self._legal_hw_from_orig(H0, W0)
        if (H_in, W_in) != (H0, W0):
            img_rgb = cv2.resize(img_rgb, (W_in, H_in), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img_rgb).permute(2,0,1).float().unsqueeze(0) / 255.0
        pm, ps = self._get_norm()
        img_t = (img_t.to(self.device, non_blocking=True) - pm) / ps
        return img_t, (H0,W0), (H_in,W_in), sy, sx

    # --- encode/decode ---
    @torch.no_grad()
    def _get_image_embed(self, img_t: torch.Tensor):
        out = self.model.image_encoder(img_t)
        if isinstance(out, dict) and ("vision_features" in out):
            vfeats = out["vision_features"]
            vpos   = out.get("vision_pos_enc", None)
            fpn    = out.get("backbone_fpn", None)

            if isinstance(vfeats, torch.Tensor): vfeats = [vfeats]
            if isinstance(vpos,   torch.Tensor): vpos   = [vpos]
            vpos_list = list(vpos) if isinstance(vpos, (list, tuple)) else []

            cand = [(i, t) for i,t in enumerate(vfeats) if torch.is_tensor(t) and t.ndim==4]
            if not cand: raise RuntimeError("vision_features has no 4D tensor")
            idx, img_feat = max(cand, key=lambda x: int(x[1].shape[-2]) * int(x[1].shape[-1]))
            Hf, Wf = int(img_feat.shape[-2]), int(img_feat.shape[-1])

            img_pe = None
            for p in vpos_list:
                if torch.is_tensor(p) and p.ndim>=3 and int(p.shape[-2])==Hf and int(p.shape[-1])==Wf:
                    img_pe = p; break

            levels = out.get("backbone_fpn", None)
            if isinstance(levels, (list, tuple)):
                levels = [x.to(self.device, non_blocking=True) for x in levels if torch.is_tensor(x)]
            elif torch.is_tensor(levels):
                levels = [levels.to(self.device, non_blocking=True)]
            else:
                levels = []
            if len(levels) >= 2:
                levels_sorted = sorted(levels, key=lambda t: int(t.shape[-2])*int(t.shape[-1]), reverse=True)
                high_res = (levels_sorted[0], levels_sorted[1])
            elif len(levels) == 1:
                high_res = (levels[0], levels[0])
            else:
                high_res = None

            img_feat = img_feat.to(self.device, non_blocking=True)
            if isinstance(img_pe, torch.Tensor):
                img_pe = img_pe.to(self.device, non_blocking=True)
            return img_feat, img_pe, high_res

        # fallback
        tensors = []
        def collect(o):
            if torch.is_tensor(o): tensors.append(o)
            elif isinstance(o, dict):
                for v in o.values(): collect(v)
            elif isinstance(o, (list,tuple)):
                for v in o: collect(v)
        collect(out)
        cand = [t for t in tensors if t.ndim==4]
        if not cand: raise RuntimeError("image_encoder returned no 4D features")
        img_feat = max(cand, key=lambda t: int(t.shape[-2])*int(t.shape[-1]))
        return img_feat.to(self.device, non_blocking=True), None, None

    @torch.no_grad()
    def _encode_prompts(self, coords: torch.Tensor, labels: torch.Tensor):
        pe = getattr(self.model, "prompt_encoder", None) or getattr(self.model, "sam_prompt_encoder", None)
        if pe is None: raise AttributeError("No prompt_encoder / sam_prompt_encoder in model")
        labels = labels.long()
        out = pe(points=(coords, labels), boxes=None, masks=None)
        if isinstance(out, (tuple, list)):
            sp, dp = out[0], out[1]
        elif isinstance(out, dict):
            sp = out.get("sparse_prompt_embeddings", out.get("points"))
            dp = out.get("dense_prompt_embeddings",  out.get("dense"))
        else:
            raise AttributeError("Unexpected prompt_encoder output type")
        if sp is not None: sp = sp.to(self.device, non_blocking=True)
        if dp is not None: dp = dp.to(self.device, non_blocking=True)
        return sp, dp

    @staticmethod
    def _match_channels(x: torch.Tensor, out_ch: int) -> torch.Tensor:
        b, c, h, w = x.shape
        if c == out_ch: return x
        if c > out_ch:  return x[:, :out_ch, :, :]
        pad = x.new_zeros((b, out_ch - c, h, w))
        return torch.cat([x, pad], dim=1)

    @torch.no_grad()
    def _probe_dc_out_channels(self, md: nn.Module, image_feat: torch.Tensor):
        def _last_conv_out_channels(mod: nn.Module) -> Optional[int]:
            last = None
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    last = m
            return int(last.out_channels) if last is not None else None
        c1 = _last_conv_out_channels(getattr(md, "dc1", nn.Identity()))
        c2 = _last_conv_out_channels(getattr(md, "dc2", nn.Identity()))
        if (c1 is not None) and (c2 is not None):
            return c1, c2
        return 64, 32

    @torch.no_grad()
    def _decode_mask(self, image_feat, image_pe, sparse_pe, dense_pe, high_res):
        md = getattr(self.model, "mask_decoder", None) or getattr(self.model, "sam_mask_decoder", None)
        if md is None: raise AttributeError("No mask_decoder / sam_mask_decoder in model")

        if dense_pe is not None and (dense_pe.shape[-2:] != image_feat.shape[-2:]):
            dense_pe = F.interpolate(dense_pe, size=image_feat.shape[-2:], mode="bilinear", align_corners=False)

        hr = None
        if isinstance(high_res, tuple) and len(high_res) == 2:
            feat_s0, feat_s1 = high_res
            tgt_s1, tgt_s0 = self._probe_dc_out_channels(md, image_feat)
            if feat_s1.shape[1] != tgt_s1: feat_s1 = self._match_channels(feat_s1, tgt_s1)
            if feat_s0.shape[1] != tgt_s0: feat_s0 = self._match_channels(feat_s0, tgt_s0)
            hr = (feat_s0, feat_s1)

        kwargs = dict(
            image_embeddings=image_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_pe,
            dense_prompt_embeddings=dense_pe,
            multimask_output=False,
            repeat_image=True,
        )
        if hr is not None:
            kwargs["high_res_features"] = hr

        out = md(**kwargs)
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict): return out.get("masks", out.get("mask_logits"))
        return out

    @torch.no_grad()
    def forward(self, images_bgr: List[np.ndarray], points_list: List[np.ndarray], metas: Optional[List[dict]] = None) -> torch.Tensor:
        feats = []
        for i, (img_bgr, pts_np) in enumerate(zip(images_bgr, points_list)):
            key = None
            if metas is not None and i < len(metas):
                p = metas[i].get("image_path", None)
                if isinstance(p, str) and len(p) > 0:
                    key = f"path::{p}"
            if key is None:
                key = f"sig::{self._img_signature(img_bgr)}"

            cached = self._cache_get(key)
            if cached is None:
                img_t, _, _, sy, sx = self._preprocess_manual(img_bgr)
                img_feat, img_pe, high_res = self._get_image_embed(img_t)
                cached = (img_feat, img_pe, high_res, sy, sx)
                self._cache_put(key, cached)
            else:
                img_feat, img_pe, high_res, sy, sx = cached

            # ---- 生成掩码（含兜底）----
            if pts_np is None or len(pts_np) == 0:
                # 无点：全图 GAP
                mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
            else:
                coords = self._map_points_scale_xy(pts_np, sy, sx).to(self.device)
                labels = torch.from_numpy(np.asarray(pts_np, np.float32)[:, 2]).unsqueeze(0).to(self.device)
                if labels.max() <= 0:
                    # 纯负点/无正点：回退到全图
                    mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
                else:
                    sp, dp = self._encode_prompts(coords, labels)
                    mask_logits = self._decode_mask(img_feat, img_pe, sp, dp, high_res)
                    mask = torch.sigmoid(mask_logits)
                    if mask.shape[-2:] != img_feat.shape[-2:]:
                        mask = F.interpolate(mask, size=img_feat.shape[-2:], mode="bilinear", align_corners=False)
                    # 面积过小或异常：再次兜底
                    if (not torch.isfinite(mask).all()) or (mask.sum() <= 1e-5):
                        mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
            # --------------------------

            feat = (img_feat * mask).flatten(2).sum(dim=-1) / (mask.flatten(2).sum(dim=-1) + 1e-6)
            feats.append(feat.squeeze(0))
        return torch.stack(feats, dim=0)


    # ---------- caching utils ----------
    def _img_signature(self, img: np.ndarray) -> str:
        h, w = img.shape[:2]
        ch = img.shape[2] if img.ndim == 3 else 1
        prefix = img.ravel()[:4096].tobytes()
        sig = hashlib.md5(prefix).hexdigest()
        return f"{h}x{w}x{ch}:{img.dtype.str}:{sig}"

    def _cache_get(self, key: str):
        v = self.cache.get(key)
        if v is not None:
            self.cache.move_to_end(key)
        return v

    def _cache_put(self, key: str, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    @staticmethod
    def _map_points_scale_xy(points: np.ndarray, sy: float, sx: float) -> torch.Tensor:
        if points is None or len(points) == 0:
            return torch.zeros((1,0,2), dtype=torch.float32)
        pts = np.asarray(points, dtype=np.float32).copy()
        pts[:, 0] *= sx; pts[:, 1] *= sy
        return torch.from_numpy(pts[:, :2]).unsqueeze(0).float()


# ----------------------------- Heads -----------------------------

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 0, drop: float = 0.0):
        super().__init__()
        if hidden and hidden > 0:
            self.fc1 = nn.Linear(in_dim, hidden)
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()
            self.fc2 = nn.Linear(hidden, n_classes)
            self._deep = True
        else:
            self.fc = nn.Linear(in_dim, n_classes)
            self._deep = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._deep: return self.fc(x)
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); return x

class MLPBNHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_layers: List[int], drop: float = 0.0):
        super().__init__()
        dims = [in_dim] + list(hidden_layers)
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), nn.GELU()]
            if drop and drop > 0: layers.append(nn.Dropout(drop))
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], n_classes)
    def forward(self, x):
        return self.out(self.mlp(x))

class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = float(scale)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x_n, w_n)


# ----------------------------- Train / Val / Test -----------------------------

def _apply_logit_adjust(logits: torch.Tensor, log_prior: Optional[torch.Tensor], tau: float):
    if (log_prior is None) or (tau is None) or (tau <= 0):
        return logits
    # 减去 τ * log(prior): 抑制高频类
    return logits - float(tau) * log_prior.view(1, -1).to(logits.device)


def train_one_epoch(extractor: nn.Module,
                    head: nn.Module,
                    loader: DataLoader,
                    device: str,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    logger: TrainingLogger,
                    loss_fn,                 # 可以是 CE 或 focal 包装
                    max_grad_norm: float = 0.0,
                    log_prior: Optional[torch.Tensor] = None,
                    logit_adjust_tau: float = 0.0):
    head.train()
    running_loss, n = 0.0, 0

    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar, 1):
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)

        with torch.no_grad():
            feats = extractor(imgs, pts, metas)

        logits = head(feats)
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
def evaluate(extractor: nn.Module,
             head: nn.Module,
             loader: DataLoader,
             device: str,
             epoch: int,
             logger: TrainingLogger,
             loss_fn,
             split_name: str = "val",
             n_classes: Optional[int] = None,
             id2name: Optional[Dict[int, str]] = None,
             log_prior: Optional[torch.Tensor] = None,
             logit_adjust_tau: float = 0.0):
    head.eval()
    total_loss, n = 0.0, 0
    correct = 0

    per_cls_total = None
    per_cls_correct = None
    if n_classes is not None:
        per_cls_total = torch.zeros(n_classes, dtype=torch.long)
        per_cls_correct = torch.zeros(n_classes, dtype=torch.long)

    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch} [{split_name}]", leave=False)
    for batch in pbar:
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)

        feats = extractor(imgs, pts, metas)
        logits = head(feats)
        logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()

        if per_cls_total is not None:
            for c in range(n_classes):
                mask = (y == c)
                if mask.any():
                    per_cls_total[c] += int(mask.sum().item())
                    per_cls_correct[c] += int((preds[mask] == c).sum().item())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)
    logger.write(f"epoch={epoch} {split_name}_loss={avg_loss:.6f} {split_name}_acc={acc:.6f}")

    if per_cls_total is not None and id2name is not None:
        print(f"[{split_name.upper()} per-class acc] (epoch {epoch})")
        for c in range(n_classes):
            n_c = int(per_cls_total[c].item())
            acc_c = (per_cls_correct[c].item() / n_c) if n_c > 0 else float('nan')
            name = id2name.get(c, f"class{c}")
            print(f"  {c:2d} {name:>16s}: acc={acc_c:.4f}  (n={n_c})")
            logger.write(f"epoch={epoch} {split_name}_clsacc[{c}][{name}]={acc_c:.6f} n={n_c}")

    return avg_loss, acc


# ----------------------------- Main -----------------------------

def _parse_hidden_list(s: str) -> List[int]:
    out = []
    for p in s.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(SMALLFILE_ROOT))
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--label-map", type=str, default=None)
    ap.add_argument("--backend", choices=["official"], default="official")
    ap.add_argument("--sam2-cfg", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_l.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_large.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--hidden", type=str, default="0")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default=str(CKPT_ROOT))
    ap.add_argument("--head", choices=["linear","mlp","mlp_bn","cosine"], default="mlp")
    ap.add_argument("--scale", type=float, default=16.0, help="scale for cosine head")
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--sched", choices=["none","cosine"], default="cosine")
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--max-grad-norm", type=float, default=0.0)
    ap.add_argument("--resume", type=str, default=None, help="path to checkpoint (e.g., best_head.pt) to resume head")
    # balancing / focal / logit-adjust
    ap.add_argument("--balance", choices=["none","weights","sampler","auto"], default="auto")
    ap.add_argument("--focal", action="store_true", help="use focal loss (gamma=2.0, alpha from class weights if available)")
    ap.add_argument("--reweight-alpha", type=float, default=0.5, help="alpha for sampling weights: w_c = n_c^(-alpha)")
    ap.add_argument("--bg-factor", type=float, default=0.3, help="downweight factor for background sampling weight")
    ap.add_argument("--logit-adjust", type=float, default=0.0, help="tau for logit adjustment: logits -= tau*log(prior)")
    # 新增：background 掩码策略
    ap.add_argument("--bg-mask-mode", choices=["pos", "global", "mix"], default="mix",
                help="background 掩码模式: pos=将BG点设为正点; global=不喂点(全图GAP); mix=二者随机混合")

    args = ap.parse_args()

    if args.resize is not None:
        print("[WARN] You set --resize. Prefer --resize=None so the wrapper handles sizing.")

    set_seed(args.seed)

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else (data_root / "manifest.csv")
    label_map_path = Path(args.label_map) if args.label_map else (data_root / "label_map.json")
    out_dir = Path(args.out_dir) if args.out_dir else CKPT_ROOT
    ensure_dir(out_dir)
    ensure_dir(SMALLFILE_ROOT)

    # Prefer existing train/val/test CSVs; otherwise create by clip_name split
    train_csv = SMALLFILE_ROOT / "train_manifest.csv"
    val_csv   = SMALLFILE_ROOT / "val_manifest.csv"
    test_csv  = SMALLFILE_ROOT / "test_manifest.csv"
    if train_csv.exists() and val_csv.exists():
        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)
        df_test  = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame(columns=df_train.columns)
    else:
        df_all = pd.read_csv(manifest_path)
        df_train, df_val, df_test = _stable_train_val_test_split(df_all, seed=args.seed,
                                                                 train_ratio=0.8, val_ratio=0.1)
        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)
        df_test.to_csv(test_csv, index=False)

    # self-check: label_map background=0
    lm = load_label_map(label_map_path)
    tool2id = lm["tool_to_id"]
    assert "background" in tool2id and tool2id["background"] == 0, \
        f"label_map.json must include 'background': 0, current={tool2id.get('background')}"
    print(f"[CHECK] tool_to_id = {tool2id}")

    # id->name map for per-class printing
    id2name = {int(v): str(k) for k, v in tool2id.items()}

    # datasets & loaders
    ds_train = FramePointDataset(train_csv, label_map_path, resize=args.resize, bg_mask_mode=args.bg_mask_mode)
    ds_val   = FramePointDataset(val_csv,   label_map_path, resize=args.resize, bg_mask_mode=args.bg_mask_mode) if len(df_val)  else None
    ds_test  = FramePointDataset(test_csv,  label_map_path, resize=args.resize, bg_mask_mode=args.bg_mask_mode) if len(df_test) else None


    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True) if ds_val else None
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True) if ds_test else None

    # class stats / priors
    counts, train_dist, imb_ratio, priors = _class_stats(train_csv, label_map_path)
    if train_dist is not None:
        print(f"[CHECK] train distribution per class = {train_dist} | imbalance ratio={imb_ratio:.2f}")
    class_weights_ce = _ce_class_weights_from_counts(counts) if counts is not None else None
    log_prior = torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32)) if priors is not None else None

    # WeightedRandomSampler（基于 ds_train.df，和 Dataset 过滤一致）
    sampler = None
    if (args.balance in ("auto","sampler")) and (counts is not None):
        use_sampler = (args.balance == "sampler") or (args.balance == "auto" and imb_ratio >= 5.0)
        if use_sampler:
            per_class_sw = _sampling_weights_from_counts(counts, alpha=args.reweight_alpha, bg_factor=args.bg_factor)
            sample_ids = [tool2id.get(t, 0) for t in ds_train.df["tool"].tolist()]
            sw = [float(per_class_sw[c]) for c in sample_ids]
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
            dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, collate_fn=collate_varlen,
                                  pin_memory=True, sampler=sampler)
            print(f"[INFO] Using WeightedRandomSampler (alpha={args.reweight_alpha}, bg_factor={args.bg_factor}).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    if args.backend == "official":
        extractor = Sam2OfficialWrapper(args.sam2_cfg, args.sam2_ckpt, device=device, cache_size=128)
    else:
        raise NotImplementedError("Only 'official' backend is provided.")

    # probe feat dim
    if len(ds_train) == 0:
        raise RuntimeError("Empty training set after filtering. Check manifest/points_json.")
    probe = next(iter(dl_train))
    with torch.no_grad():
        feat = extractor(probe["images"][:1], probe["points"][:1], probe["meta"][:1])
    in_dim = int(feat.shape[-1])
    n_classes = len(tool2id)

    # build head
    hidden_list = _parse_hidden_list(args.hidden)
    if args.head == "linear":
        head = MLPHead(in_dim, n_classes, hidden=0, drop=args.drop).to(device)
    elif args.head == "mlp":
        h = hidden_list[0] if hidden_list else 0
        head = MLPHead(in_dim, n_classes, hidden=h, drop=args.drop).to(device)
    elif args.head == "mlp_bn":
        if not hidden_list:
            hidden_list = [1024, 512]
        head = MLPBNHead(in_dim, n_classes, hidden_layers=hidden_list, drop=args.drop).to(device)
    else:  # cosine
        head = CosineClassifier(in_dim, n_classes, scale=args.scale).to(device)

    # optimizer / scheduler / loss
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.focal:
        gamma = 2.0
        alpha = class_weights_ce.to(device) if class_weights_ce is not None else None
        def focal_loss(logits, target):
            ce = F.cross_entropy(logits, target, reduction="none", weight=alpha)
            pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0)
            fl = ((1 - pt) ** gamma) * ce
            return fl.mean()
        loss_fn = focal_loss
        print("[INFO] Using Focal Loss (gamma=2).")
    else:
        if (args.balance in ("weights","auto")) and (class_weights_ce is not None) and (imb_ratio is not None and imb_ratio >= 5.0):
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_ce.to(device), label_smoothing=args.smoothing)
            print("[INFO] Using CE with class weights.")
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    if args.sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.01)
    else:
        sched = None

    # resume head
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("head_state", ckpt)
        try:
            head.load_state_dict(state, strict=True)
            print(f"[RESUME] Loaded head weights from {args.resume}")
        except Exception as e:
            print(f"[RESUME] Strict load failed: {e}\nTrying non-strict...")
            head.load_state_dict(state, strict=False)
            print(f"[RESUME] Non-strict loaded from {args.resume}")

    # logger
    log_file = SMALLFILE_ROOT / "train_log.txt"
    logger = TrainingLogger(log_file)

    try:
        best_acc = -1.0
        best_epoch = -1
        patience_left = args.patience
        best_path = out_dir / "best_head.pt"

        for epoch in range(1, args.epochs + 1):
            # warmup
            if args.warmup_epochs and epoch <= args.warmup_epochs:
                warmup_ratio = epoch / max(1, args.warmup_epochs)
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * (0.1 + 0.9 * warmup_ratio)

            tr_loss = train_one_epoch(extractor, head, dl_train, device, opt, epoch, logger,
                                      loss_fn=loss_fn, max_grad_norm=args.max_grad_norm,
                                      log_prior=log_prior, logit_adjust_tau=args.logit_adjust)
            va_loss, va_acc = evaluate(extractor, head, dl_val, device, epoch, logger,
                                       loss_fn=loss_fn, split_name="val",
                                       n_classes=n_classes, id2name=id2name,
                                       log_prior=log_prior, logit_adjust_tau=args.logit_adjust) if dl_val else (0.0, 0.0)

            print(f"[{epoch:02d}] train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} val_acc {va_acc:.3f}")

            if sched is not None and (not args.warmup_epochs or epoch > args.warmup_epochs):
                sched.step()

            improved = (dl_val is None) or (va_acc > best_acc)
            if improved:
                best_acc = va_acc
                best_epoch = epoch
                patience_left = args.patience
                torch.save({
                    "head_state": head.state_dict(),
                    "in_dim": in_dim,
                    "n_classes": n_classes,
                    "tool_to_id": tool2id,
                    "args": vars(args),
                }, str(best_path))
                logger.write(f"epoch={epoch} SAVED best_head -> {best_path}")
            else:
                patience_left -= 1

            if (epoch % 5 == 0) or (epoch == args.epochs):
                ep_path = out_dir / f"head_epoch{epoch:03d}.pt"
                torch.save({
                    "head_state": head.state_dict(),
                    "in_dim": in_dim,
                    "n_classes": n_classes,
                    "tool_to_id": tool2id,
                    "args": vars(args),
                }, str(ep_path))
                logger.write(f"epoch={epoch} SAVED periodic_head -> {ep_path}")

            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best val acc={best_acc:.4f} (epoch {best_epoch}).")
                logger.write(f"early_stop best_acc={best_acc:.6f} best_epoch={best_epoch}")
                break

        # optional test with per-class acc too
        if dl_test:
            test_loss, test_acc = evaluate(extractor, head, dl_test, device, epoch=best_epoch, logger=logger,
                                           loss_fn=loss_fn, split_name="test",
                                           n_classes=n_classes, id2name=id2name,
                                           log_prior=log_prior, logit_adjust_tau=args.logit_adjust)
            print(f"[TEST] loss {test_loss:.4f} acc {test_acc:.3f}")
            logger.write(f"test_loss={test_loss:.6f} test_acc={test_acc:.6f}")

        print(f"Done. Best val acc={best_acc:.4f} at epoch {best_epoch}. Saved head to: {best_path}")
        logger.write(f"done best_acc={best_acc:.6f} best_epoch={best_epoch} path={best_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

# CE+logit adjust
# python /home/wcheng31/sam2_classify/train_sam2_classify.py \
#   --backend official \
#   --epochs 20 --batch-size 256 --seed 123 --patience 5 \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --balance auto --reweight-alpha 0.5 --bg-factor 0.7 \
#   --bg-mask-mode mix --logit-adjust 0.5

# Focal，不用 logit adjust
# python /home/wcheng31/sam2_classify/train_sam2_classify.py \
#   --backend official \
#   --epochs 20 --batch-size 256 --seed 123 --patience 5 \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --balance auto --reweight-alpha 0.5 --bg-factor 0.7 \
#   --bg-mask-mode mix --focal


#   --resume /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/best_head.pt
