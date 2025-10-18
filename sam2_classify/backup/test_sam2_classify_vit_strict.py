#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格评测（与 train/evaluate() 对齐）：
- Sam2OfficialWrapper(images, points, metas, return_4d=True) -> (img_feat, mask, img_pe)
- ViTTokenHead(img_feat, mask, img_pe) -> logits
- 应用 logit-adjust（Balanced Softmax）
- 计算 loss/acc、混淆矩阵 + per-class Precision/Recall/F1，并保存混淆矩阵图
- 可选导出预测 CSV

仅评测，不做训练。
"""

import os, json, argparse, time, math, hashlib
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
    def tqdm(x, **k): return x

# ----------------- 常量路径 -----------------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"

# ----------------- Hydra 初始化（与训练一致） -----------------
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

def setup_hydra_configs():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path="configs/sam2", version_base="1.2")

# ---------- 使用 backup 构建器 ----------
from sam2.backup.build_sam import build_sam2

# =========================================================
# Utils / Dataset / Collate
# =========================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_label_map(p: Path) -> Dict[str, Dict[str, int]]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

class FramePointDataset(Dataset):
    def __init__(self, manifest_csv: Path, label_map_json: Path, resize: Optional[int] = None, bg_mask_mode: str = "mix"):
        super().__init__()
        self.df = pd.read_csv(manifest_csv)
        self.label_map = load_label_map(label_map_json)
        self.tool2id = self.label_map["tool_to_id"]
        self.resize = resize
        self.bg_mask_mode = bg_mask_mode

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
        H0, W0 = img.shape[:2]
        if self.resize and self.resize > 0:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        return img, (H0, W0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img, (H0, W0) = self._load_img(row["image_path"])
        H1, W1 = img.shape[:2]

        pts = json.loads(row["points_json"]) if isinstance(row["points_json"], str) and row["points_json"].strip() else []
        pts_np = np.asarray(pts, np.float32) if len(pts) else np.zeros((0, 3), np.float32)

        if (H0 != H1) or (W0 != W1):
            sy = float(H1) / max(1.0, float(H0))
            sx = float(W1) / max(1.0, float(W0))
            if pts_np.size > 0:
                pts_np[:, 0] *= sx; pts_np[:, 1] *= sy

        pts_out = []
        for p in pts_np:
            if len(p) < 2: continue
            x = float(np.clip(p[0], 0, W1 - 1)); y = float(np.clip(p[1], 0, H1 - 1))
            label = 1.0 if (len(p) >= 3 and float(p[2]) > 0) else 0.0
            pts_out.append([x, y, label])

        tool = str(row["tool"])
        tool_id = self.tool2id.get(tool, None)
        if tool_id is None:
            for k, v in self.tool2id.items():
                if str(k) == tool: tool_id = v; break
        if tool_id is None: raise KeyError(f"Tool '{tool}' not in label_map.json")

        # 背景样本点策略与训练一致（mix 兜底）
        if int(tool_id) == 0:
            import random
            if self.bg_mask_mode == "pos":
                for p in pts_out: p[2] = 1.0
            elif self.bg_mask_mode == "global":
                pts_out = []
            else:
                if random.random() < 0.5: pts_out = []
                else:
                    for p in pts_out: p[2] = 1.0

        return {
            "image": img,
            "points": np.array(pts_out, dtype=np.float32) if pts_out else np.zeros((0, 3), np.float32),
            "tool_id": int(tool_id),
            "meta": {
                "image_path": row["image_path"],
                "tool": tool,
            }
        }

def collate_varlen(batch):
    images  = [b["image"]  for b in batch]
    points  = [b["points"] for b in batch]
    targets = torch.tensor([b["tool_id"] for b in batch], dtype=torch.long)
    metas   = [b["meta"]   for b in batch]
    return {"images": images, "points": points, "targets": targets, "meta": metas}

# =========================================================
# Sam2OfficialWrapper（复刻训练脚本关键行为）
# =========================================================
class Sam2OfficialWrapper(nn.Module):
    def __init__(self, cfg: str, ckpt: str, device: str = "cuda", cache_size: int = 128):
        super().__init__()
        self.device = device
        setup_hydra_configs()
        self.model = build_sam2(cfg, ckpt, device=device)
        self.model.to(self.device); self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self._norm_cached: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.cache: "OrderedDict[str, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], float, float]]" = OrderedDict()
        self.cache_size = int(cache_size)
        self.verbose = False; self._printed_resize = False

    # ---- infer legal size ----
    def _infer_patch_conv(self):
        enc = getattr(self.model, "image_encoder", None)
        trunk = getattr(enc, "trunk", None)
        if trunk is None or not hasattr(trunk, "patch_embed") or not hasattr(trunk.patch_embed, "proj"):
            return 7,7,4,4,3,3
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
        t_h = max(win, (t_h0 // win) * win); t_w = max(win, (t_w0 // win) * win)
        H_in = self._size_for_tokens(t_h, k_h, s_h, p_h); W_in = self._size_for_tokens(t_w, k_w, s_w, p_w)
        sy = H_in / max(1, H0); sx = W_in / max(1, W0)
        if self.verbose and not self._printed_resize:
            print(f"[SAM2] orig=({H0},{W0}) -> in=({H_in},{W_in})"); self._printed_resize = True
        return H_in, W_in, sy, sx

    # ---- normalization ----
    def _get_norm(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._norm_cached is not None: return self._norm_cached
        for obj in [self.model, getattr(self.model, "image_encoder", None)]:
            pm = getattr(obj, "pixel_mean", None); ps = getattr(obj, "pixel_std",  None)
            if pm is not None and ps is not None:
                pm = torch.as_tensor(pm, dtype=torch.float32).view(1,3,1,1)
                ps = torch.as_tensor(ps, dtype=torch.float32).view(1,3,1,1)
                if pm.max() > 1.5 or ps.max() > 1.5: pm = pm/255.0; ps = ps/255.0
                self._norm_cached = (pm.to(self.device), ps.to(self.device)); return self._norm_cached
        pm = torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1)
        ps = torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1)
        self._norm_cached = (pm,ps); return self._norm_cached

    @torch.no_grad()
    def _preprocess_manual(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0,W0 = img_rgb.shape[:2]
        H_in,W_in,sy,sx = self._legal_hw_from_orig(H0, W0)
        if (H_in,W_in)!=(H0,W0): img_rgb = cv2.resize(img_rgb, (W_in,H_in), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img_rgb).permute(2,0,1).float().unsqueeze(0)/255.0
        pm,ps = self._get_norm()
        img_t = (img_t.to(self.device, non_blocking=True) - pm)/ps
        return img_t, (H0,W0), (H_in,W_in), sy, sx

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
            cand = [(i,t) for i,t in enumerate(vfeats) if torch.is_tensor(t) and t.ndim==4]
            if not cand: raise RuntimeError("vision_features has no 4D tensor")
            _, img_feat = max(cand, key=lambda x: int(x[1].shape[-2])*int(x[1].shape[-1]))
            Hf,Wf = int(img_feat.shape[-2]), int(img_feat.shape[-1])
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
            if len(levels)>=2:
                levels_sorted = sorted(levels, key=lambda t: int(t.shape[-2])*int(t.shape[-1]), reverse=True)
                high_res = (levels_sorted[0], levels_sorted[1])
            elif len(levels)==1:
                high_res = (levels[0], levels[0])
            else:
                high_res = None
            img_feat = img_feat.to(self.device, non_blocking=True)
            if isinstance(img_pe, torch.Tensor): img_pe = img_pe.to(self.device, non_blocking=True)
            return img_feat, img_pe, high_res
        # fallback: 收集 4D tensor
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
        if pe is None: raise AttributeError("No prompt_encoder in model")
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
        b,c,h,w = x.shape
        if c == out_ch: return x
        if c > out_ch:  return x[:, :out_ch, :, :]
        pad = x.new_zeros((b, out_ch-c, h, w))
        return torch.cat([x, pad], dim=1)

    @torch.no_grad()
    def _probe_dc_out_channels(self, md: nn.Module, image_feat: torch.Tensor):
        def _last_conv_out_channels(mod: nn.Module) -> Optional[int]:
            last = None
            for m in mod.modules():
                if isinstance(m, nn.Conv2d): last = m
            return int(last.out_channels) if last is not None else None
        c1 = _last_conv_out_channels(getattr(md, "dc1", nn.Identity()))
        c2 = _last_conv_out_channels(getattr(md, "dc2", nn.Identity()))
        if (c1 is not None) and (c2 is not None): return c1, c2
        return 64, 32

    @torch.no_grad()
    def _decode_mask(self, image_feat, image_pe, sparse_pe, dense_pe, high_res):
        md = getattr(self.model, "mask_decoder", None) or getattr(self.model, "sam_mask_decoder", None)
        if md is None: raise AttributeError("No mask_decoder in model")
        if dense_pe is not None and (dense_pe.shape[-2:] != image_feat.shape[-2:]):
            dense_pe = F.interpolate(dense_pe, size=image_feat.shape[-2:], mode="bilinear", align_corners=False)
        hr = None
        if isinstance(high_res, tuple) and len(high_res)==2:
            feat_s0, feat_s1 = high_res
            tgt_s1, tgt_s0 = self._probe_dc_out_channels(md, image_feat)
            if feat_s1.shape[1] != tgt_s1: feat_s1 = self._match_channels(feat_s1, tgt_s1)
            if feat_s0.shape[1] != tgt_s0: feat_s0 = self._match_channels(feat_s0, tgt_s0)
            hr = (feat_s0, feat_s1)
        kwargs = dict(image_embeddings=image_feat, image_pe=image_pe,
                      sparse_prompt_embeddings=sparse_pe, dense_prompt_embeddings=dense_pe,
                      multimask_output=False, repeat_image=True)
        if hr is not None: kwargs["high_res_features"] = hr
        out = md(**kwargs)
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict): return out.get("masks", out.get("mask_logits"))
        return out

    @torch.no_grad()
    def _align_batch_4d(self, feats_triplets):
        sizes = [t[0].shape[-2:] for t in feats_triplets]
        tgt_h = min(h for h, _ in sizes)
        tgt_w = min(w for _, w in sizes)

        def _pool_to(x, size_hw):
            if x is None: return None
            if x.shape[-2:] == size_hw: return x
            return F.adaptive_avg_pool2d(x, size_hw)

        img_feats, masks, pos = [], [], []
        for (ff, mm, pp) in feats_triplets:
            ff_r = _pool_to(ff, (tgt_h, tgt_w))
            mm_r = _pool_to(mm, (tgt_h, tgt_w))
            if pp is None:
                pp_r = torch.zeros_like(ff_r)
            else:
                pp_r = _pool_to(pp, (tgt_h, tgt_w))
            img_feats.append(ff_r); masks.append(mm_r); pos.append(pp_r)

        img_feat_b = torch.cat(img_feats, dim=0)
        mask_b     = torch.cat(masks,    dim=0)
        pos_b      = torch.cat(pos,      dim=0)
        return img_feat_b, mask_b, pos_b

    @torch.no_grad()
    def forward(self, images_bgr: List[np.ndarray], points_list: List[np.ndarray],
                metas: Optional[List[dict]] = None, return_4d: bool = False):
        feats_triplets = []
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

            # 生成掩码（与训练一致：点全负或无点 -> 全1兜底）
            if pts_np is None or len(pts_np) == 0:
                mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
            else:
                coords = self._map_points_scale_xy(pts_np, sy, sx).to(self.device)
                labels = torch.from_numpy(np.asarray(pts_np, np.float32)[:, 2]).unsqueeze(0).to(self.device)
                if labels.max() <= 0:
                    mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
                else:
                    sp, dp = self._encode_prompts(coords, labels)
                    mask_logits = self._decode_mask(img_feat, img_pe, sp, dp, high_res)
                    mask = torch.sigmoid(mask_logits)
                    if mask.shape[-2:] != img_feat.shape[-2:]:
                        mask = F.interpolate(mask, size=img_feat.shape[-2:], mode="bilinear", align_corners=False)
                    if (not torch.isfinite(mask).all()) or (mask.sum() <= 1e-5):
                        mask = torch.ones((1, 1, img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
            feats_triplets.append((img_feat, mask, img_pe))

        if return_4d:
            img_feat_b, mask_b, pos_b = self._align_batch_4d(feats_triplets)
            return img_feat_b, mask_b, pos_b

        pooled = []
        for (img_feat, mask, _) in feats_triplets:
            feat = (img_feat * mask).flatten(2).sum(dim=-1) / (mask.flatten(2).sum(dim=-1) + 1e-6)
            pooled.append(feat.squeeze(0))
        return torch.stack(pooled, dim=0)

    # ---------- cache utils ----------
    def _img_signature(self, img: np.ndarray) -> str:
        h, w = img.shape[:2]
        ch = img.shape[2] if img.ndim == 3 else 1
        prefix = img.ravel()[:4096].tobytes()
        sig = hashlib.md5(prefix).hexdigest()
        return f"{h}x{w}x{ch}:{img.dtype.str}:{sig}"

    def _cache_get(self, key: str):
        v = self.cache.get(key)
        if v is not None: self.cache.move_to_end(key)
        return v

    def _cache_put(self, key: str, value):
        self.cache[key] = value; self.cache.move_to_end(key)
        if len(self.cache) > self.cache_size: self.cache.popitem(last=False)

    @staticmethod
    def _map_points_scale_xy(points: np.ndarray, sy: float, sx: float) -> torch.Tensor:
        if points is None or len(points) == 0: return torch.zeros((1,0,2), dtype=torch.float32)
        pts = np.asarray(points, dtype=np.float32).copy()
        pts[:, 0] *= sx; pts[:, 1] *= sy
        return torch.from_numpy(pts[:, :2]).unsqueeze(0).float()

# =========================================================
# 头部：ViT 轻量头（与训练一致）
# =========================================================
class ViTTokenHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int,
                 num_layers: int = 2, num_heads: int = 4,
                 mlp_ratio: float = 4.0, p_drop: float = 0.05,
                 max_tokens: int = 1024):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.cls = nn.Parameter(torch.zeros(1, 1, in_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads,
            dim_feedforward=int(mlp_ratio*in_dim),
            dropout=p_drop, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(in_dim)
        self.fc   = nn.Linear(in_dim, n_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, img_feat: torch.Tensor, mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        B,C,H,W = img_feat.shape
        N = H * W
        if N > self.max_tokens:
            s = math.sqrt(N / float(self.max_tokens))
            Ht = max(1, int(H / s + 0.5))
            Wt = max(1, int(W / s + 0.5))
            while Ht * Wt > self.max_tokens:
                if Ht >= Wt and Ht > 1: Ht -= 1
                elif Wt > 1:            Wt -= 1
                else:                   break
            img_feat = F.adaptive_avg_pool2d(img_feat, (Ht, Wt))
            if pos is not None:
                pos = F.adaptive_avg_pool2d(pos, (Ht, Wt))
            if mask is not None:
                mask = F.adaptive_max_pool2d(mask, (Ht, Wt))
            H, W = Ht, Wt

        x = img_feat if pos is None else (img_feat + pos)
        x = x.permute(0,2,3,1).reshape(B, H*W, C)

        key_padding = None
        if mask is not None:
            thr = 0.3
            keep = (mask > thr).flatten(1)
            Kmin = min(64, H*W)
            for i in range(B):
                if int(keep[i].sum().item()) < Kmin:
                    vals = mask[i,0].flatten()
                    k = min(Kmin, vals.numel())
                    topk = torch.topk(vals, k=k, dim=0).indices
                    keep[i].zero_(); keep[i, topk] = True
            key_padding = (~keep).bool()

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if key_padding is not None:
            pad0 = torch.zeros(B,1, dtype=torch.bool, device=x.device)
            key_padding = torch.cat([pad0, key_padding], dim=1)

        x = self.enc(x, src_key_padding_mask=key_padding)
        cls_out = self.norm(x[:,0])
        return self.fc(cls_out)

# =========================================================
# 评测核心：evaluate_strict（等价于训练 evaluate）
# =========================================================
def _save_confmat_figure(cm: np.ndarray, id2name: Dict[int,str], save_path: Path, title: str):
    if not HAS_MPL:
        print(f"[WARN] matplotlib not available, skip saving {save_path}")
        return
    ensure_dir(save_path.parent)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum>0)
    fig, ax = plt.subplots(figsize=(6,5), dpi=180)
    im = ax.imshow(cm_norm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = [id2name.get(i, str(i)) for i in range(cm.shape[0])]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, ylabel="GT", xlabel="Pred", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), va="center", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)

def _apply_logit_adjust(logits: torch.Tensor, log_prior: Optional[torch.Tensor], tau: float):
    if (log_prior is None) or (tau is None) or (tau <= 0): return logits
    return logits - float(tau) * log_prior.view(1, -1).to(logits.device)

@torch.no_grad()
def evaluate_strict(extractor: nn.Module, head: nn.Module, loader: DataLoader, device: str,
                    loss_fn, id2name: Dict[int,str], log_prior: Optional[torch.Tensor],
                    logit_adjust_tau: float, out_cm_path: Path,
                    save_preds_path: Optional[Path] = None):
    """严格复刻训练 evaluate() 的逻辑；固定 split_name='test' epoch=0"""
    torch.cuda.empty_cache()
    head.eval()
    total_loss, n = 0.0, 0
    correct = 0
    printed_shapes = False

    n_classes = len(id2name)
    cm = torch.zeros((n_classes, n_classes), dtype=torch.long)
    preds_records = []

    pbar = tqdm(loader, total=len(loader), ncols=100, desc="Epoch 0 [test]", leave=False)
    for batch in pbar:
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)

        img_feat, mask, img_pe = extractor(imgs, pts, metas, return_4d=True)
        if not printed_shapes:
            B,C,H,W = img_feat.shape
            out_dim = head.fc.out_features if hasattr(head,"fc") else None
            print(f"[SHAPE][E0][test] ViT in=({B},{C},{H},{W}) out=({y.size(0)},{out_dim})")
            printed_shapes = True

        logits = head(img_feat, mask, img_pe)
        logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()

        # 混淆矩阵
        y_cpu = y.view(-1).to("cpu")
        p_cpu = preds.view(-1).to("cpu")
        idx = y_cpu * n_classes + p_cpu
        cm += torch.bincount(idx, minlength=n_classes*n_classes).view(n_classes, n_classes)

        if save_preds_path is not None:
            probs = torch.softmax(logits, dim=1).to("cpu").numpy()
            for i in range(bs):
                rec = {
                    "image_path": metas[i].get("image_path",""),
                    "tool": metas[i].get("tool",""),
                    "gt_id": int(y_cpu[i].item()),
                    "pred_id": int(p_cpu[i].item()),
                    "pred_name": id2name.get(int(p_cpu[i].item()), str(int(p_cpu[i].item()))),
                    "conf": float(probs[i, p_cpu[i]].item()),
                }
                preds_records.append(rec)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)
    print(f"[TEST] epoch 0  loss={avg_loss:.4f}  acc={acc:.4f}")

    # per-class P/R/F1
    cm_np = cm.numpy()
    tp = np.diag(cm_np)
    gt_per_cls = cm_np.sum(axis=1)    # 行
    pred_per_cls = cm_np.sum(axis=0)  # 列
    eps = 1e-12
    recall = np.divide(tp, gt_per_cls + eps)
    precision = np.divide(tp, pred_per_cls + eps)
    f1 = np.where((precision+recall) < eps, 0.0, 2*precision*recall/(precision+recall))

    print("[TEST per-class metrics] (epoch 0)")
    for c in range(n_classes):
        name = id2name.get(c, f"class{c}")
        print(f"  {c:2d} {name:>16s}: P={precision[c]:.4f} R={recall[c]:.4f} F1={f1[c]:.4f} (GT={int(gt_per_cls[c])}, Pred={int(pred_per_cls[c])})")

    valid_gt = gt_per_cls > 0
    valid_pred = pred_per_cls > 0
    macro_recall = float(recall[valid_gt].mean()) if valid_gt.any() else float("nan")
    macro_precision = float(precision[valid_pred].mean()) if valid_pred.any() else float("nan")
    macro_f1 = 0.0 if (macro_precision + macro_recall) < eps else 2*macro_precision*macro_recall/(macro_precision+macro_recall)
    print(f"[TEST macro] (epoch 0): P={macro_precision:.4f} R={macro_recall:.4f} F1={macro_f1:.4f}")

    # 保存混淆矩阵图
    _save_confmat_figure(cm_np, id2name, out_cm_path, title="Confusion Matrix [test] epoch 0")
    print(f"[TEST] confusion matrix saved to: {out_cm_path}")

    # 导出预测
    if save_preds_path is not None and len(preds_records):
        dfp = pd.DataFrame(preds_records)
        ensure_dir(save_preds_path.parent)
        dfp.to_csv(save_preds_path, index=False, encoding="utf-8")
        print(f"[TEST] predictions saved to: {save_preds_path}")

    return avg_loss, acc

# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(SMALLFILE_ROOT))
    ap.add_argument("--manifest", type=str, default=None, help="默认使用 config/test_manifest_10.csv；也可给你的 test.csv")
    ap.add_argument("--label-map", type=str, default=None)

    ap.add_argument("--backend", choices=["official"], default="official")
    ap.add_argument("--sam2-cfg",  type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_t.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_tiny.pt"))

    ap.add_argument("--ckpt", type=str, required=True, help="训练时保存的 ViT 头 ckpt（best_head.pt/head_epochXXX.pt）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--bg-mask-mode", choices=["pos","global","mix"], default="mix")

    # ViT 轻量头参数（如 ckpt 有 args 则优先）
    ap.add_argument("--vit-layers", type=int, default=2)
    ap.add_argument("--vit-heads",  type=int, default=4)
    ap.add_argument("--vit-drop",   type=float, default=0.05)
    ap.add_argument("--vit-mlp-ratio", type=float, default=4.0)
    ap.add_argument("--vit-max-tokens", type=int, default=1024)

    # logit-adjust（Balanced Softmax）
    ap.add_argument("--logit-adjust", type=float, default=1.0)

    # 输出
    ap.add_argument("--save-preds", type=str, default=None, help="可选：保存预测CSV路径")
    ap.add_argument("--out-cm", type=str, default=str(SMALLFILE_ROOT / "cm_test_epoch000.png"))

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = "cuda"; torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else (data_root / "test_manifest_10.csv")
    label_map_path = Path(args.label_map) if args.label_map else (data_root / "label_map.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found: {manifest_path}")

    # label_map
    lm = load_label_map(label_map_path)
    tool2id = lm["tool_to_id"]
    assert "background" in tool2id and tool2id["background"] == 0
    id2name = {int(v): str(k) for k, v in tool2id.items()}

    # dataset & loader
    ds_test  = FramePointDataset(manifest_path, label_map_path, resize=args.resize, bg_mask_mode=args.bg_mask_mode)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True)

    # extractor
    if args.backend != "official":
        raise NotImplementedError("Only 'official' backend is provided.")
    extractor = Sam2OfficialWrapper(args.sam2_cfg, args.sam2_ckpt, device=device, cache_size=256)

    # 从 ckpt 恢复 ViT 头（超参以 ckpt.args 优先）
    ckpt = torch.load(args.ckpt, map_location="cpu")
    in_dim = int(ckpt.get("in_dim"))  # 训练时记录的 C
    n_classes = int(ckpt.get("n_classes"))
    ck_args = ckpt.get("args", {}) or {}

    vit_layers = ck_args.get("vit_layers", args.vit_layers)
    vit_heads  = ck_args.get("vit_heads",  args.vit_heads)
    vit_drop   = ck_args.get("vit_drop",   args.vit_drop)
    vit_mlp    = ck_args.get("vit_mlp_ratio", args.vit_mlp_ratio if hasattr(args,"vit_mlp_ratio") else 4.0)
    vit_max_tk = ck_args.get("vit_max_tokens", args.vit_max_tokens)

    head = ViTTokenHead(in_dim=in_dim, n_classes=n_classes,
                        num_layers=int(vit_layers), num_heads=int(vit_heads),
                        mlp_ratio=float(vit_mlp), p_drop=float(vit_drop),
                        max_tokens=int(vit_max_tk)).to(device)
    state = ckpt.get("head_state", ckpt)
    head.load_state_dict(state, strict=False)
    head.eval()

    # logit priors（优先使用 ckpt 内的 priors/counts；否则均匀）
    priors = None
    if "priors" in ckpt:
        priors = np.asarray(ckpt["priors"], dtype=np.float32)
    elif "counts" in ckpt and np.asarray(ckpt["counts"]).sum() > 0:
        counts = np.asarray(ckpt["counts"], dtype=np.float32)
        priors = counts / counts.sum()
    if priors is None:
        priors = np.ones(n_classes, dtype=np.float32) / float(n_classes)
    log_prior = torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32)).to(device)

    loss_fn = nn.CrossEntropyLoss()

    out_cm_path = Path(args.out_cm)
    save_preds_path = Path(args.save_preds) if args.save_preds else None

    evaluate_strict(
        extractor=extractor, head=head, loader=dl_test, device=device,
        loss_fn=loss_fn, id2name=id2name, log_prior=log_prior,
        logit_adjust_tau=float(args.logit_adjust), out_cm_path=out_cm_path,
        save_preds_path=save_preds_path
    )

if __name__ == "__main__":
    main()


# python /home/wcheng31/sam2_classify/test_sam2_classify_vit_strict.py \
#   --data-root /home/wcheng31/sam2_classify/config \
#   --manifest /home/wcheng31/sam2_classify/config/test_manifest_10.csv \
#   --label-map /home/wcheng31/sam2_classify/config/label_map.json \
#   --sam2-cfg /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_t.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_tiny.pt \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/vit_head/best_head.pt \
#   --batch-size 128 --workers 8 \
#   --bg-mask-mode mix \
#   --logit-adjust 1.0 \
#   --out-cm /home/wcheng31/sam2_classify/config/cm_test_epoch000.png \
#   --save-preds /home/wcheng31/sam2_classify/config/test_preds.csv
