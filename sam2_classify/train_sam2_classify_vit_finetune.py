#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ---- tqdm ----
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

# ---- plotting (optional) ----
HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    HAS_MPL = False

# ----------------- Hydra 初始化 -----------------
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

def setup_hydra_configs():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path="configs/sam2", version_base="1.2")

# ---------- 使用 backup 构建器 ----------
from sam2.backup.build_sam import build_sam2

# ---------- 路径 ----------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"

# ---------- Utils ----------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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
        try: self.f.close()
        except Exception: pass

def _class_stats(train_csv: Path, label_map_json: Path):
    lm = load_label_map(label_map_json)
    tool2id = lm["tool_to_id"]
    df = pd.read_csv(train_csv)
    ids = [tool2id[t] for t in df["tool"] if t in tool2id]
    if len(ids) == 0: return None, None, None, None
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
    if len(w) > 0: w[0] *= float(bg_factor)
    return w

# ---------- Dataset ----------
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

        # 背景点策略
        if int(tool_id) == 0:
            mode = self.bg_mask_mode
            if mode == "pos":
                for p in pts_out: p[2] = 1.0
            elif mode == "global":
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
                "frame_abs_index": int(row.get("frame_abs_index", -1)),
                "frame_idx_in_clip": int(row.get("frame_idx_in_clip", -1)),
                "clip_name": row.get("clip_name", ""),
                "task": row.get("task", ""),
                "orig_hw": (int(H0), int(W0)),
                "used_hw": (int(H1), int(W1)),
            }
        }

def collate_varlen(batch):
    images  = [b["image"]  for b in batch]
    points  = [b["points"] for b in batch]
    targets = torch.tensor([b["tool_id"] for b in batch], dtype=torch.long)
    metas   = [b["meta"]   for b in batch]
    return {"images": images, "points": points, "targets": targets, "meta": metas}

# ---------- SAM2 Wrapper (可训练) ----------
class Sam2OfficialWrapper(nn.Module):
    def __init__(self, cfg: str, ckpt: str, device: str = "cuda", cache_size: int = 128):
        super().__init__()
        self.device = device
        setup_hydra_configs()
        self.model = build_sam2(cfg, ckpt, device=device)
        self.model.to(self.device)
        for p in self.model.parameters(): p.requires_grad_(True)
        self.model.train()

        self.verbose = False
        self._printed_resize = False
        self._norm_cached: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.cache: "OrderedDict[str, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], float, float]]" = OrderedDict()
        self.cache_size = int(cache_size)

    # ---- size utils ----
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

    def _preprocess_manual(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0,W0 = img_rgb.shape[:2]
        H_in,W_in,sy,sx = self._legal_hw_from_orig(H0, W0)
        if (H_in,W_in)!=(H0,W0): img_rgb = cv2.resize(img_rgb, (W_in,H_in), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img_rgb).permute(2,0,1).float().unsqueeze(0)/255.0
        pm,ps = self._get_norm()
        img_t = (img_t.to(self.device, non_blocking=True) - pm)/ps
        return img_t, (H0,W0), (H_in,W_in), sy, sx

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
            if pp is None: pp_r = torch.zeros_like(ff_r)
            else: pp_r = _pool_to(pp, (tgt_h, tgt_w))
            img_feats.append(ff_r); masks.append(mm_r); pos.append(pp_r)
        img_feat_b = torch.cat(img_feats, dim=0)
        mask_b     = torch.cat(masks,    dim=0)
        pos_b      = torch.cat(pos,      dim=0)
        return img_feat_b, mask_b, pos_b

    def forward(self, images_bgr: List[np.ndarray], points_list: List[np.ndarray],
                metas: Optional[List[dict]] = None, return_4d: bool = False):
        feats_triplets = []
        for i, (img_bgr, pts_np) in enumerate(zip(images_bgr, points_list)):
            use_cache = (not self.training) and (self.cache_size > 0)

            key = None
            if metas is not None and i < len(metas):
                p = metas[i].get("image_path", None)
                if isinstance(p, str) and len(p) > 0:
                    key = f"path::{p}"
            if key is None:
                key = f"sig::{self._img_signature(img_bgr)}"

            cached = self._cache_get(key) if use_cache else None
            if cached is None:
                img_t, _, _, sy, sx = self._preprocess_manual(img_bgr)
                img_feat, img_pe, high_res = self._get_image_embed(img_t)
                if use_cache:
                    self._cache_put(key, (img_feat, img_pe, high_res, sy, sx))
            else:
                img_feat, img_pe, high_res, sy, sx = cached

            # ---- 生成掩码 ----
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
        return f"{h}xw{w}x{ch}:{img.dtype.str}:{sig}"

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

# ---------- Heads ----------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 0, drop: float = 0.0):
        super().__init__()
        if hidden and hidden > 0:
            self.fc1 = nn.Linear(in_dim, hidden); self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()
            self.fc2 = nn.Linear(hidden, n_classes); self._deep = True
        else:
            self.fc = nn.Linear(in_dim, n_classes); self._deep = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._deep: return self.fc(x)
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); return x

class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_normal_(self.weight); self.scale = float(scale)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, dim=1); w_n = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x_n, w_n)

class ViTTokenHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int,
                 num_layers: int = 2, num_heads: int = 4,
                 mlp_ratio: float = 4.0, p_drop: float = 0.05,
                 max_tokens: int = 1024, min_keep_tokens: int = 256):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.min_keep_tokens = int(min_keep_tokens)
        self.last_token_count = None  # 记录本次 forward 的 token 数（H*W）
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

    def _build_sincos_pos(self, B, C, H, W, device):
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yy = yy / max(1.0, H); xx = xx / max(1.0, W)
        freqs = [1.0, 2.0, 4.0, 8.0]
        pe_list = []
        for f in freqs:
            pe_list += [torch.sin(2*math.pi*f*yy), torch.cos(2*math.pi*f*yy),
                        torch.sin(2*math.pi*f*xx), torch.cos(2*math.pi*f*xx)]
        pe = torch.stack(pe_list, dim=-1)
        if pe.shape[-1] < C:
            pad = torch.zeros(H, W, C - pe.shape[-1], device=device)
            pe = torch.cat([pe, pad], dim=-1)
        elif pe.shape[-1] > C:
            pe = pe[..., :C]
        return pe.permute(2,0,1).unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    def forward(self, img_feat: torch.Tensor, mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        B,C,H,W = img_feat.shape
        # 限顶 tokens
        N = H * W
        if N > self.max_tokens:
            s = math.sqrt(N / float(self.max_tokens))
            Ht = max(1, int(H / s + 0.5))
            Wt = max(1, int(W / s + 0.5))
            while Ht * Wt > self.max_tokens:
                if Ht >= Wt and Ht > 1: Ht -= 1
                elif Wt > 1:            Wt -= 1
                else: break
            img_feat = F.adaptive_avg_pool2d(img_feat, (Ht, Wt))
            if pos is not None:  pos  = F.adaptive_avg_pool2d(pos,  (Ht, Wt))
            if mask is not None: mask = F.adaptive_max_pool2d(mask, (Ht, Wt))
            H, W = Ht, Wt
        self.last_token_count = H * W  # <<< 记录送入 Transformer 的 token 个数（不含 cls）

        if pos is None:
            pos = self._build_sincos_pos(B, C, H, W, img_feat.device)

        x = (img_feat + pos).permute(0,2,3,1).reshape(B, H*W, C)

        key_padding = None
        if mask is not None:
            thr = 0.3
            keep = (mask > thr).flatten(1)          # [B,N]
            Kmin = min(self.min_keep_tokens, H*W)
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

# ---------- Train / Eval ----------
def _apply_logit_adjust(logits: torch.Tensor, log_prior: Optional[torch.Tensor], tau: float):
    if (log_prior is None) or (tau is None) or (tau <= 0): return logits
    return logits - float(tau) * log_prior.view(1, -1).to(logits.device)

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

def train_one_epoch(extractor: nn.Module, head: nn.Module, loader: DataLoader, device: str,
                    optimizer: torch.optim.Optimizer, epoch: int, logger: TrainingLogger, loss_fn,
                    max_grad_norm: float = 0.0, log_prior: Optional[torch.Tensor] = None,
                    logit_adjust_tau: float = 0.0, head_type: str = "vit"):
    extractor.train()
    head.train()
    running_loss, n = 0.0, 0
    printed_shapes = False
    printed_tokens = False

    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar, 1):
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)

        if head_type == "vit":
            img_feat, mask, img_pe = extractor(imgs, pts, metas, return_4d=True)
            feats_for_head = (img_feat, mask, img_pe)
        else:
            vecs = extractor(imgs, pts, metas, return_4d=False)
            feats_for_head = vecs

        if not printed_shapes:
            if head_type == "vit":
                B,C,H,W = feats_for_head[0].shape
                logger.write(f"epoch={epoch} [shapes/train] x=[{B},{C},{H},{W}] -> logits=[{y.size(0)},{head.fc.out_features}]")
                print(f"[SHAPE][E{epoch}][train] ViT in=({B},{C},{H},{W})  out=({y.size(0)},{head.fc.out_features})")
            else:
                B,C = feats_for_head.shape
                out_dim = head.weight.shape[0] if hasattr(head,'weight') else head.fc.out_features
                logger.write(f"epoch={epoch} [shapes/train] x=[{B},{C}] -> logits=[{y.size(0)},{out_dim}]")
                print(f"[SHAPE][E{epoch}][train] GAP in=({B},{C})  out=({y.size(0)},{out_dim})")
            printed_shapes = True

        if head_type == "vit":
            img_feat, mask, img_pe = feats_for_head
            logits = head(img_feat, mask, img_pe)
            if (not printed_tokens) and hasattr(head, "last_token_count"):
                print(f"[TOKENS][E{epoch}] ViT tokens = {head.last_token_count} (per sample, excl. [CLS])")
                logger.write(f"epoch={epoch} tokens={head.last_token_count}")
                printed_tokens = True
        else:
            logits = head(feats_for_head)

        logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(extractor.parameters()) + list(head.parameters()), max_grad_norm)
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
             log_prior: Optional[torch.Tensor] = None, logit_adjust_tau: float = 0.0, head_type: str = "vit",
             save_dir: Optional[Path] = None):
    torch.cuda.empty_cache()
    extractor.eval()
    head.eval()
    total_loss, n = 0.0, 0
    correct = 0
    printed_shapes = False
    printed_tokens = False

    cm = None
    if n_classes is not None:
        cm = torch.zeros((n_classes, n_classes), dtype=torch.long)

    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch} [{split_name}]", leave=False)
    for batch in pbar:
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)

        if head_type == "vit":
            img_feat, mask, img_pe = extractor(imgs, pts, metas, return_4d=True)
            if not printed_shapes:
                B,C,H,W = img_feat.shape
                print(f"[SHAPE][E{epoch}][{split_name}] ViT in=({B},{C},{H},{W}) out=({y.size(0)},{head.fc.out_features})")
                logger.write(f"epoch={epoch} [shapes/{split_name}] x=[{B},{C},{H},{W}] -> logits=[{y.size(0)},{head.fc.out_features}]")
                printed_shapes = True
            logits = head(img_feat, mask, img_pe)
            if (not printed_tokens) and hasattr(head, "last_token_count"):
                print(f"[TOKENS][E{epoch}][{split_name}] ViT tokens = {head.last_token_count} (per sample, excl. [CLS])")
                logger.write(f"epoch={epoch} {split_name}_tokens={head.last_token_count}")
                printed_tokens = True
        else:
            vecs = extractor(imgs, pts, metas, return_4d=False)
            if not printed_shapes:
                B,C = vecs.shape
                out_dim = head.weight.shape[0] if hasattr(head,'weight') else head.fc.out_features
                print(f"[SHAPE][E{epoch}][{split_name}] GAP in=({B},{C}) out=({y.size(0)},{out_dim})")
                logger.write(f"epoch={epoch} [shapes/{split_name}] x=[{B},{C}] -> logits=[{y.size(0)},{out_dim}]")
                printed_shapes = True
            logits = head(vecs)

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
            cm += torch.bincount(idx, minlength=n_classes*n_classes).view(n_classes, n_classes)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)
    logger.write(f"epoch={epoch} {split_name}_loss={avg_loss:.6f} {split_name}_acc={acc:.6f}")
    print(f"[{split_name.upper()}] epoch {epoch}  loss={avg_loss:.4f}  acc={acc:.4f}")

    # 简单保存混淆矩阵图（可选） + per-class 指标
    if (cm is not None) and (id2name is not None):
        cm_np = cm.numpy()

        # per-class metrics
        tp = np.diag(cm_np)
        gt_per_cls = cm_np.sum(axis=1)
        pred_per_cls = cm_np.sum(axis=0)
        eps = 1e-12
        recall = np.divide(tp, gt_per_cls + eps)
        precision = np.divide(tp, pred_per_cls + eps)
        f1 = np.where((precision+recall) < eps, 0.0, 2*precision*recall/(precision+recall))

        print(f"[{split_name.upper()} per-class metrics] (epoch {epoch})")
        for c in range(cm_np.shape[0]):
            name = id2name.get(c, f"class{c}")
            print(f"  {c:2d} {name:>16s}: P={precision[c]:.4f} R={recall[c]:.4f} F1={f1[c]:.4f} (GT={int(gt_per_cls[c])}, Pred={int(pred_per_cls[c])})")
            logger.write(f"epoch={epoch} {split_name}_percls[{c}][{name}] P={precision[c]:.6f} R={recall[c]:.6f} F1={f1[c]:.6f} GT={int(gt_per_cls[c])} Pred={int(pred_per_cls[c])}")

        save_base = (save_dir if save_dir is not None else SMALLFILE_ROOT)
        ensure_dir(save_base)
        save_png = Path(save_base) / f"cm_{split_name}_epoch{epoch:03d}.png"
        _save_confmat_figure(cm_np, id2name, save_png, title=f"Confusion Matrix [{split_name}] epoch {epoch}")
        print(f"[{split_name.upper()}] confusion matrix saved to: {save_png}")

        # 也保存原始 cm 到 csv
        save_csv = Path(save_base) / f"cm_{split_name}_epoch{epoch:03d}.csv"
        pd.DataFrame(cm_np, index=[id2name.get(i, str(i)) for i in range(cm_np.shape[0])],
                           columns=[id2name.get(i, str(i)) for i in range(cm_np.shape[1])]).to_csv(save_csv)
        print(f"[{split_name.upper()}] confusion matrix CSV saved to: {save_csv}")

    return avg_loss, acc

@torch.no_grad()
def evaluate_and_dump(extractor: nn.Module,
                      head: nn.Module,
                      loader: DataLoader,
                      device: str,
                      n_classes: int,
                      id2name: Dict[int, str],
                      save_dir: Path,
                      log_prior: Optional[torch.Tensor] = None,
                      logit_adjust_tau: float = 0.0,
                      head_type: str = "vit"):
    """
    评测并将结果落盘：
      - confusion_matrix.png / confusion_matrix.csv
      - predictions.csv （逐样本）
      - summary.txt （总体与分类别指标）
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    all_targets, all_preds, all_probs, img_paths = [], [], [], []

    extractor.eval()
    head.eval()
    pbar = tqdm(loader, total=len(loader), ncols=100, desc="TEST", leave=False)
    for batch in pbar:
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)

        if head_type == "vit":
            img_feat, mask, img_pe = extractor(imgs, pts, metas, return_4d=True)
            logits = head(img_feat, mask, img_pe)
        else:
            vecs = extractor(imgs, pts, metas, return_4d=False)
            logits = head(vecs)

        logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        all_targets.append(y.cpu())
        all_preds.append(pred.cpu())
        all_probs.append(prob.cpu())
        img_paths.extend([m.get("image_path", "") for m in metas])

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    prob   = torch.cat(all_probs).numpy()

    # 混淆矩阵（整数计数）
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    # 保存图像版混淆矩阵
    _save_confmat_figure(cm, id2name, save_dir / "confusion_matrix.png", title="Confusion Matrix [test]")

    # 保存 CSV 版混淆矩阵
    pd.DataFrame(
        cm,
        index=[id2name.get(i, str(i)) for i in range(n_classes)],
        columns=[id2name.get(i, str(i)) for i in range(n_classes)]
    ).to_csv(save_dir / "confusion_matrix.csv")

    # 逐样本预测
    top1_prob = prob[np.arange(len(prob)), y_pred]
    pd.DataFrame({
        "image_path": img_paths,
        "true_id": y_true,
        "true_name": [id2name.get(int(t), str(int(t))) for t in y_true],
        "pred_id": y_pred,
        "pred_name": [id2name.get(int(p), str(int(p))) for p in y_pred],
        "pred_conf": top1_prob
    }).to_csv(save_dir / "predictions.csv", index=False)

    # 分类别指标 + 总体
    tp = np.diag(cm).astype(float)
    gt_per_cls = cm.sum(axis=1).astype(float)
    pred_per_cls = cm.sum(axis=0).astype(float)
    eps = 1e-12
    recall = np.divide(tp, gt_per_cls + eps)
    precision = np.divide(tp, pred_per_cls + eps)
    f1 = np.where((precision + recall) < eps, 0.0, 2 * precision * recall / (precision + recall))
    overall_acc = (y_true == y_pred).mean()

    with open(save_dir / "summary.txt", "w") as f:
        f.write(f"overall_acc={overall_acc:.6f}\n")
        for cid in range(n_classes):
            name = id2name.get(cid, f"class{cid}")
            f.write(
                f"cls[{cid}][{name}] P={precision[cid]:.6f} R={recall[cid]:.6f} "
                f"F1={f1[cid]:.6f} GT={int(gt_per_cls[cid])} Pred={int(pred_per_cls[cid])}\n"
            )

    print(f"[TEST] overall_acc={overall_acc:.4f} | results saved to: {save_dir}")
    return overall_acc, cm


# ---------- Main ----------
def _parse_hidden_list(s: str) -> List[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(SMALLFILE_ROOT))
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--label-map", type=str, default=None)
    ap.add_argument("--backend", choices=["official"], default="official")

    # 默认 sam2 tiny
    ap.add_argument("--sam2-cfg",  type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_t.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_tiny.pt"))

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--hidden", type=str, default="0")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="lr for SAM2 backbone")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default=str(CKPT_ROOT))

    ap.add_argument("--head", choices=["linear","cosine","vit"], default="vit")
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--sched", choices=["none","cosine"], default="cosine")
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--max-grad-norm", type=float, default=0.0)
    ap.add_argument("--resume", type=str, default=None)

    # ViT 头
    ap.add_argument("--vit-layers", type=int, default=2)
    ap.add_argument("--vit-heads",  type=int, default=4)
    ap.add_argument("--vit-drop",   type=float, default=0.05)
    ap.add_argument("--vit-mlp-ratio", type=float, default=4.0)
    ap.add_argument("--vit-max-tokens", type=int, default=1024)
    ap.add_argument("--vit-min-keep",  type=int, default=256)

    # class balance
    ap.add_argument("--balance", choices=["none","weights","sampler","auto"], default="auto")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--reweight-alpha", type=float, default=1.0)
    ap.add_argument("--bg-factor", type=float, default=0.1)
    ap.add_argument("--logit-adjust", type=float, default=1.0)

    # eval/test batch size
    ap.add_argument("--val-batch-size", type=int, default=None)
    ap.add_argument("--test-batch-size", type=int, default=None)

    # ===================== 新增：测试模式与数据集切换 =====================
    ap.add_argument("--test-only", action="store_true",
                    help="等价于 --mode test")
    ap.add_argument("--test-dataset", choices=["combined","cholec80"], default=None,
                    help="优先于 --dataset；未提供时沿用 --dataset")
    ap.add_argument("--cholec80-root", type=str, default=None,
                    help="若提供，将在该目录下寻找 manifest.csv / label_map.json")
    ap.add_argument("--combined-manifest", type=str,
                    default=str(SMALLFILE_ROOT / "test_manifest_10.csv"))
    ap.add_argument("--combined-labelmap", type=str,
                    default=str(SMALLFILE_ROOT / "label_map.json"))
    # 在 argparse 定义处（和 --test-only 等放一起）
    ap.add_argument(
        "--save-root",
        type=str,
        default=str(CKPT_ROOT / "eval_vit_e2e"),
        help="测试结果输出目录（将写入 confusion_matrix.(png|csv), predictions.csv, summary.txt）",
    )
    # ====================================================================

    args = ap.parse_args()
    if args.resize is not None:
        print("[WARN] You set --resize. Prefer --resize=None so wrapper handles sizing.")

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = "cuda"; torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else (data_root / "manifest.csv")
    label_map_path = Path(args.label_map) if args.label_map else (data_root / "label_map.json")
    out_dir = Path(args.out_dir) if args.out_dir else CKPT_ROOT
    if args.head == "vit": out_dir = out_dir / "vit_head_e2e"
    ensure_dir(out_dir); ensure_dir(SMALLFILE_ROOT)

    # ===================== TEST 分支（不改训练流程） =====================
    if args.test_only or args.mode == "test":
        # 1) 解析数据集与路径
        test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        if test_dataset == "combined":
            test_manifest_path = Path(args.combined_manifest)
            test_labelmap_path = Path(args.combined_labelmap)
        else:  # cholec80
            if args.cholec80_root is not None:
                root = Path(args.cholec80_root)
                test_manifest_path = root / "manifest.csv"
                test_labelmap_path = root / "label_map.json"
            else:
                test_manifest_path = Path(args.cholec80_manifest)
                test_labelmap_path = Path(args.cholec80_labelmap)

        if not test_manifest_path.exists():
            raise FileNotFoundError(f"Test manifest not found: {test_manifest_path}")
        if not test_labelmap_path.exists():
            raise FileNotFoundError(f"Test label_map not found: {test_labelmap_path}")

        # 2) label map / 类别映射
        lm_eval = load_label_map(test_labelmap_path)
        tool2id_eval = lm_eval["tool_to_id"]
        assert "background" in tool2id_eval and tool2id_eval["background"] == 0
        id2name_eval = {int(v): str(k) for k, v in tool2id_eval.items()}

        # 3) dataloader
        ds_eval = FramePointDataset(test_manifest_path, test_labelmap_path, resize=args.resize, bg_mask_mode="mix")
        if len(ds_eval) == 0:
            raise RuntimeError("Empty test set after filtering. Check test manifest/points_json.")
        dl_eval = DataLoader(ds_eval,
                             batch_size=(args.test_batch_size or args.batch_size),
                             shuffle=False, num_workers=args.workers,
                             collate_fn=collate_varlen, pin_memory=True)

        # 4) 构建模型 + 还原权重（E2E：SAM2 + ViT 头）
        if args.backend != "official":
            raise NotImplementedError("Only 'official' backend is provided.")
        extractor = Sam2OfficialWrapper(args.sam2_cfg, args.sam2_ckpt, device=device, cache_size=128)

        if args.resume is None or (not Path(args.resume).exists()):
            raise FileNotFoundError("--resume checkpoint missing for test mode")
        ckpt = torch.load(args.resume, map_location="cpu")
        in_dim = int(ckpt.get("in_dim", 0))
        n_classes = int(ckpt.get("n_classes", len(tool2id_eval)))

        if in_dim <= 0:
            probe = next(iter(dl_eval))
            with torch.no_grad():
                img_feat_probe, _, _ = extractor(probe["images"][:1], probe["points"][:1], probe["meta"][:1], return_4d=True)
                in_dim = int(img_feat_probe.shape[1])

        head = ViTTokenHead(in_dim=in_dim, n_classes=n_classes,
                            num_layers=args.vit_layers, num_heads=args.vit_heads,
                            mlp_ratio=args.vit_mlp_ratio, p_drop=args.vit_drop,
                            max_tokens=args.vit_max_tokens, min_keep_tokens=args.vit_min_keep).to(device)

        # 恢复权重
        hs = ckpt.get("head_state", None)
        if hs is not None:
            try:
                head.load_state_dict(hs, strict=True)
            except Exception:
                head.load_state_dict(hs, strict=False)
        ss = ckpt.get("sam2_state", None)
        if ss is not None:
            try:
                extractor.model.load_state_dict(ss, strict=True)
            except Exception:
                extractor.model.load_state_dict(ss, strict=False)

        # 5) 保存目录（保持稳定路径，便于自动化收集）
        save_dir = Path(args.save_root) / test_dataset / "unfreeze_sam2_VIT"
        ensure_dir(save_dir)

        print(f"Dataset: {test_dataset} ({test_manifest_path})")
        print(f"Model: SAM2({Path(args.sam2_cfg).name}) + ViT head | checkpoint={args.resume}")
        print(f"Save to: {save_dir}")

        # 6) 评测并落盘（与 distill 版统一导出）
        #   先验通常不以 test 分布为准，默认关闭；若要开启可改为 torch.log(torch.tensor(priors+1e-12))
        _ = evaluate_and_dump(
            extractor, head, dl_eval, device,
            n_classes=n_classes, id2name=id2name_eval,
            save_dir=save_dir,
            log_prior=None, logit_adjust_tau=args.logit_adjust,
            head_type="vit"
        )
        return

    # ===================== TEST 分支结束 =====================

    # 使用现成 split
    train_csv = SMALLFILE_ROOT / "train_manifest_10.csv"
    val_csv   = SMALLFILE_ROOT / "val_manifest_10.csv"
    test_csv  = SMALLFILE_ROOT / "test_manifest_10.csv"
    if not (train_csv.exists() and val_csv.exists()):
        raise FileNotFoundError("train/val manifest not found under SMALLFILE_ROOT")

    # label_map check
    lm = load_label_map(label_map_path)
    tool2id = lm["tool_to_id"]
    assert "background" in tool2id and tool2id["background"] == 0
    id2name = {int(v): str(k) for k, v in tool2id.items()}

    # datasets & loaders
    ds_train = FramePointDataset(train_csv, label_map_path, resize=args.resize, bg_mask_mode="mix")
    ds_val   = FramePointDataset(val_csv,   label_map_path, resize=args.resize, bg_mask_mode="mix")
    ds_test  = FramePointDataset(test_csv,  label_map_path, resize=args.resize, bg_mask_mode="mix") if test_csv.exists() else None

    train_bs = args.batch_size
    val_bs   = args.val_batch_size  if args.val_batch_size  is not None else args.batch_size
    test_bs  = args.test_batch_size if args.test_batch_size is not None else args.batch_size

    dl_train = DataLoader(ds_train, batch_size=train_bs, shuffle=True,  num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=val_bs,   shuffle=False, num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=test_bs,  shuffle=False, num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True) if ds_test else None

    # class stats
    counts, train_dist, imb_ratio, priors = _class_stats(train_csv, label_map_path)
    if train_dist is not None:
        print(f"[CHECK] train distribution per class = {train_dist} | imbalance ratio={imb_ratio:.2f}")
    class_weights_ce = _ce_class_weights_from_counts(counts) if counts is not None else None
    log_prior = torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32)) if priors is not None else None

    # sampler（可选）
    sampler = None
    if (args.balance in ("auto","sampler")) and (counts is not None):
        use_sampler = (args.balance == "sampler") or (args.balance == "auto" and imb_ratio >= 5.0)
        if use_sampler:
            per_class_sw = _sampling_weights_from_counts(counts, alpha=args.reweight_alpha, bg_factor=args.bg_factor)
            sample_ids = [tool2id.get(t, 0) for t in ds_train.df["tool"].tolist()]
            sw = [float(per_class_sw[c]) for c in sample_ids]
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
            dl_train = DataLoader(ds_train, batch_size=train_bs, shuffle=False, num_workers=args.workers,
                                  collate_fn=collate_varlen, pin_memory=True, sampler=sampler)
            print(f"[INFO] Using WeightedRandomSampler (alpha={args.reweight_alpha}, bg_factor={args.bg_factor}).")

    # model
    if args.backend != "official":
        raise NotImplementedError("Only 'official' backend is provided.")
    extractor = Sam2OfficialWrapper(args.sam2_cfg, args.sam2_ckpt, device=device, cache_size=128)

    # probe dims
    if len(ds_train) == 0:
        raise RuntimeError("Empty training set.")
    probe = next(iter(dl_train))
    with torch.no_grad():
        img_feat_probe, mask_probe, img_pe_probe = extractor(probe["images"][:1], probe["points"][:1], probe["meta"][:1], return_4d=True)
        C = int(img_feat_probe.shape[1]); in_dim = C
        H1, W1 = int(img_feat_probe.shape[-2]), int(img_feat_probe.shape[-1])
        N1 = H1 * W1
        auto_max_tokens = max(args.vit_max_tokens, min(4096, int(1.25 * N1)))
        print(f"[PROBE] img_feat={tuple(img_feat_probe.shape)} mask={tuple(mask_probe.shape)} pos={'None' if img_pe_probe is None else tuple(img_pe_probe.shape)}")
        print(f"[VIT] auto_max_tokens={auto_max_tokens} (from N={N1})")

    n_classes = len(tool2id)

    # build head
    if args.head == "vit":
        head = ViTTokenHead(in_dim=in_dim, n_classes=n_classes,
                            num_layers=args.vit_layers, num_heads=args.vit_heads,
                            mlp_ratio=args.vit_mlp_ratio, p_drop=args.vit_drop,
                            max_tokens=auto_max_tokens, min_keep_tokens=args.vit_min_keep).to(device)
        print(f"[VIT] max_tokens={auto_max_tokens}  min_keep_tokens={args.vit_min_keep}")
    elif args.head == "linear":
        head = MLPHead(in_dim, n_classes, hidden=0, drop=args.drop).to(device)
    else:
        head = CosineClassifier(in_dim, n_classes, scale=args.scale).to(device)

    # optimizer / scheduler / loss（两组参数：backbone + head）
    opt = torch.optim.AdamW(
        [
            {"params": extractor.parameters(), "lr": args.backbone_lr, "weight_decay": args.weight_decay},
            {"params": head.parameters(),      "lr": args.lr,          "weight_decay": args.weight_decay},
        ]
    )

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
        if (not using_sampler) and (args.balance in ("weights","auto")) and (class_weights_ce is not None) and (imb_ratio is not None and imb_ratio >= 5.0):
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_ce.to(device), label_smoothing=args.smoothing)
            print("[INFO] Using CE with class weights.")
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.01) if args.sched=="cosine" else None

    # resume（同时恢复 SAM2 和 head）
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        hs = ckpt.get("head_state", None)
        if hs is not None:
            try:
                head.load_state_dict(hs, strict=True)
                print(f"[RESUME] Loaded head from {args.resume}")
            except Exception as e:
                print(f"[RESUME] Head strict load failed: {e}\nTrying non-strict...")
                head.load_state_dict(hs, strict=False)
        ss = ckpt.get("sam2_state", None)
        if ss is not None:
            try:
                extractor.model.load_state_dict(ss, strict=True)
                print(f"[RESUME] Loaded SAM2 from {args.resume}")
            except Exception as e:
                print(f"[RESUME] SAM2 strict load failed: {e}\nTrying non-strict...")
                extractor.model.load_state_dict(ss, strict=False)

    # logger
    log_file = SMALLFILE_ROOT / "train_log_vit_e2e.txt"
    logger = TrainingLogger(log_file)

    try:
        best_acc = -1.0; best_epoch = -1
        patience_left = args.patience
        best_path = out_dir / "best_e2e.pt"
        best_sam2_path = out_dir / "best_sam2.pt"

        for epoch in range(1, args.epochs + 1):
            if args.warmup_epochs and epoch <= args.warmup_epochs:
                warmup_ratio = epoch / max(1, args.warmup_epochs)
                for pg in opt.param_groups:
                    base_lr = args.backbone_lr if pg["params"] and next(iter(pg["params"])).requires_grad and pg is opt.param_groups[0] else args.lr
                    pg["lr"] = base_lr * (0.1 + 0.9 * warmup_ratio)

            tr_loss = train_one_epoch(extractor, head, dl_train, device, opt, epoch, logger,
                                      loss_fn=loss_fn, max_grad_norm=args.max_grad_norm,
                                      log_prior=torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32)).to(device) if args.logit_adjust>0 and priors is not None else None,
                                      logit_adjust_tau=args.logit_adjust, head_type=args.head)
            va_loss, va_acc = evaluate(extractor, head, dl_val, device, epoch, logger,
                                       loss_fn=loss_fn, split_name="val",
                                       n_classes=len(tool2id), id2name={int(v):k for k,v in tool2id.items()},
                                       log_prior=torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32)).to(device) if args.logit_adjust>0 and priors is not None else None,
                                       logit_adjust_tau=args.logit_adjust, head_type=args.head)

            print(f"[{epoch:02d}] train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} val_acc {va_acc:.3f}")
            if sched is not None and (not args.warmup_epochs or epoch > args.warmup_epochs): sched.step()

            improved = (va_acc > best_acc)
            if improved:
                best_acc = va_acc; best_epoch = epoch; patience_left = args.patience
                payload = {
                    "sam2_state": extractor.model.state_dict(),
                    "head_state": head.state_dict(),
                    "in_dim": in_dim,
                    "n_classes": len(tool2id),
                    "tool_to_id": tool2id,
                    "args": vars(args),
                }
                torch.save(payload, str(best_path))
                torch.save({"sam2_state": extractor.model.state_dict(), "args": vars(args)}, str(best_sam2_path))
                logger.write(f"epoch={epoch} SAVED best -> {best_path} | sam2 -> {best_sam2_path}")
            else:
                patience_left -= 1

            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best val acc={best_acc:.4f} (epoch {best_epoch}).")
                logger.write(f"early_stop best_acc={best_acc:.6f} best_epoch={best_epoch}")
                break

        if dl_test:
            test_loss, test_acc = evaluate(extractor, head, dl_test, device, epoch=best_epoch, logger=logger,
                                           loss_fn=loss_fn, split_name="test",
                                           n_classes=len(tool2id), id2name={int(v):k for k,v in tool2id.items()},
                                           log_prior=torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32)).to(device) if args.logit_adjust>0 and priors is not None else None,
                                           logit_adjust_tau=args.logit_adjust, head_type=args.head)
            print(f"[TEST] loss {test_loss:.4f} acc {test_acc:.3f}")
            logger.write(f"test_loss={test_loss:.6f} test_acc={test_acc:.6f}")

        print(f"Done. Best val acc={best_acc:.4f} at epoch {best_epoch}. Saved to: {best_path}")
        logger.write(f"done best_acc={best_acc:.6f} best_epoch={best_epoch} path={best_path}")
    finally:
        logger.close()

if __name__ == "__main__":
    main()


# python /home/wcheng31/sam2_classify/train_sam2_classify_vit_finetune.py \
#   --backend official \
#   --epochs 20 --batch-size 16 --val-batch-size 64 \
#   --seed 123 --patience 5 \
#   --sam2-cfg  sam2_hiera_t.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_tiny.pt \
#   --head vit --vit-layers 2 --vit-heads 4 --vit-drop 0.05 \
#   --vit-max-tokens 1024 --vit-min-keep 256 \
#   --lr 1e-3 --backbone-lr 1e-4 \
#   --logit-adjust 1.0 \
#   --balance sampler --focal --reweight-alpha 1.0 --bg-factor 0.1



# python /home/wcheng31/sam2_classify/train_sam2_classify_vit_finetune.py \
#   --test-only \
#   --test-dataset cholec80 \
#   --cholec80-root /projects/surgical-video-digital-twin/Wenzheng/IPCAI/cholec80_test41-80 \
#   --backend official \
#   --sam2-cfg sam2_hiera_t.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_tiny.pt \
#   --head vit --vit-layers 2 --vit-heads 4 --vit-drop 0.05 \
#   --vit-max-tokens 1024 --vit-min-keep 256 \
#   --test-batch-size 64 \
#   --resume /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/vit_head_e2e/best_e2e.pt \
#   --save-root /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/eval_vit_e2e
