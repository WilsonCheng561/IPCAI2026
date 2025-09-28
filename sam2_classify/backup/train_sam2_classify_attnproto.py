#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案A：掩码引导的注意力池化 + CosFace度量头（SAM2 冻结）
- 冻结 SAM2，只读出 (C,H',W') 与按点提示得到的 mask(1,H',W')
- 用“掩码引导注意力池化（K个可学习query）”聚合空间特征，比 GAP 更鲁棒
- 头部用 CosFace（可选 margin），或切回 MLP/Linear
- 类别均衡：WeightedRandomSampler / Class-Balanced CE / Logit-Adjust（三选一或组合）
- 每个 epoch 训练与验证（带 tqdm），保存 best ckpt 与分周期 ckpt
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
    def tqdm(x, **k): return x

# ----------------------------- Global paths -----------------------------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")    # csv、日志等
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

def _stable_train_val_test_split(df: pd.DataFrame, seed: int = 42,
                                 train_ratio: float = 0.8, val_ratio: float = 0.1):
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for _, g in df.groupby("clip_name"):
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

# ---- Class stats / balancing ----
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
        self.bg_mask_mode = bg_mask_mode

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
        return img

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

        # BG 策略（与之前一致）
        if int(tool_id) == 0:
            mode = self.bg_mask_mode
            if mode == "pos":
                for p in pts_out: p[2] = 1.0
            elif mode == "global":
                pts_out = []
            else:  # mix
                if random.random() < 0.5: pts_out = []
                else:
                    for p in pts_out: p[2] = 1.0

        return {
            "image": img,
            "points": np.array(pts_out, dtype=np.float32) if pts_out else np.zeros((0,3), np.float32),
            "tool_id": int(tool_id),
            "meta": {
                "image_path": row["image_path"],
                "tool": tool,
                "clip_name": row.get("clip_name", "")
            }
        }

def collate_varlen(batch):
    images  = [b["image"]  for b in batch]
    points  = [b["points"] for b in batch]
    targets = torch.tensor([b["tool_id"] for b in batch], dtype=torch.long)
    metas   = [b["meta"]   for b in batch]
    return {"images": images, "points": points, "targets": targets, "meta": metas}

# ----------------------------- SAM2 Frozen Wrapper（复用你之前的写法，略精简） -----------------------------
class Sam2OfficialWrapper(nn.Module):
    """
    冻结版 SAM2 封装：
    - encode_with_mask(images, points) -> (B,C,H',W'), (B,1,H',W')
    - 始终提供 high_res_features 给 mask_decoder，避免 None 触发的解包报错
    - 自动通道对齐以匹配 mask_decoder 的期望通道数
    """
    def __init__(self, cfg: str, ckpt: str, device: str = "cuda", cache_size: int = 128):
        super().__init__()
        self.device = device
        from sam2.build_sam import build_sam2
        self.model = build_sam2(cfg, ckpt, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._norm_cached: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.cache: "OrderedDict[str, tuple]" = OrderedDict()
        self.cache_size = int(cache_size)

    # ---------- 低层几何/规范化 ----------
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
        return 16

    def _tokens_for(self, n_pix: int, k: int, s: int, p: int) -> int:
        return int((n_pix + 2*p - k) // s + 1)
    def _size_for_tokens(self, n_tok: int, k: int, s: int, p: int) -> int:
        return int(s * (n_tok - 1) + k - 2 * p)

    def _legal_hw_from_orig(self, H0: int, W0: int):
        k_h,k_w,s_h,s_w,p_h,p_w = self._infer_patch_conv()
        win = self._infer_window_size()
        t_h0 = max(1, self._tokens_for(H0, k_h, s_h, p_h))
        t_w0 = max(1, self._tokens_for(W0, k_w, s_w, p_w))
        t_h = max(win, (t_h0 // win) * win)
        t_w = max(win, (t_w0 // win) * win)
        H_in = self._size_for_tokens(t_h, k_h, s_h, p_h)
        W_in = self._size_for_tokens(t_w, k_w, s_w, p_w)
        sy = H_in / max(1, H0); sx = W_in / max(1, W0)
        return H_in, W_in, sy, sx

    def _get_norm(self):
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

    # ---------- 通道/高分辨率辅助 ----------
    @staticmethod
    def _match_channels(x: torch.Tensor, out_ch: int) -> torch.Tensor:
        b, c, h, w = x.shape
        if c == out_ch: return x
        if c > out_ch:  return x[:, :out_ch, :, :]
        pad = x.new_zeros((b, out_ch - c, h, w))
        return torch.cat([x, pad], dim=1)

    @torch.no_grad()
    def _probe_dc_out_channels(self, md: nn.Module, image_feat: torch.Tensor) -> Tuple[int,int]:
        # 粗略探测 mask_decoder 两个解码分支最后一层 conv 的 out_channels
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
        # 兜底：常见默认
        return 64, 32

    # ---------- 编码 ----------
    @torch.no_grad()
    def _get_image_embed(self, img_t: torch.Tensor):
        """
        返回：img_feat (1,C,Hf,Wf), img_pe 或 None, high_res=(feat_s0, feat_s1) 一定非 None
        """
        out = self.model.image_encoder(img_t)

        tensors_4d = []
        def _collect(o):
            if torch.is_tensor(o):
                tensors_4d.append(o)
            elif isinstance(o, dict):
                for v in o.values(): _collect(v)
            elif isinstance(o, (list, tuple)):
                for v in o: _collect(v)

        # 1) 主特征 & 位置编码
        img_feat, img_pe = None, None
        high_res = None

        if isinstance(out, dict):
            vfeats = out.get("vision_features", None)
            if isinstance(vfeats, torch.Tensor): vfeats = [vfeats]
            if isinstance(vfeats, (list, tuple)):
                cand = [(i, t) for i,t in enumerate(vfeats) if torch.is_tensor(t) and t.ndim==4]
                if len(cand):
                    _, img_feat = max(cand, key=lambda x: int(x[1].shape[-2])*int(x[1].shape[-1]))
                    Hf, Wf = int(img_feat.shape[-2]), int(img_feat.shape[-1])
                    vpos = out.get("vision_pos_enc", None)
                    if isinstance(vpos, (list, tuple)):
                        for p in vpos:
                            if torch.is_tensor(p) and p.ndim>=3 and int(p.shape[-2])==Hf and int(p.shape[-1])==Wf:
                                img_pe = p; break
                    elif torch.is_tensor(vpos) and int(vpos.shape[-2])==Hf and int(vpos.shape[-1])==Wf:
                        img_pe = vpos

            # 2) high_res: backbone_fpn（首选两个最高分辨率）
            levels = out.get("backbone_fpn", None)
            feats = []
            if isinstance(levels, torch.Tensor):
                feats = [levels]
            elif isinstance(levels, (list, tuple)):
                feats = [t for t in levels if torch.is_tensor(t) and t.ndim==4]

            if len(feats) >= 2:
                feats_sorted = sorted(feats, key=lambda t: int(t.shape[-2])*int(t.shape[-1]), reverse=True)
                high_res = (feats_sorted[0], feats_sorted[1])
            elif len(feats) == 1:
                high_res = (feats[0], feats[0])

        # 3) fallback：遍历所有输出拿到最大空间分辨率的 4D tensor
        if img_feat is None:
            _collect(out)
            cand = [t for t in tensors_4d if t.ndim==4]
            if not cand:
                raise RuntimeError("image_encoder returned no 4D feature maps")
            img_feat = max(cand, key=lambda t: int(t.shape[-2])*int(t.shape[-1]))

        # 4) 如果 high_res 仍然 None，用 img_feat 自举两份
        if high_res is None:
            high_res = (img_feat, img_feat)

        img_feat = img_feat.to(self.device, non_blocking=True)
        if torch.is_tensor(img_pe): img_pe = img_pe.to(self.device, non_blocking=True)
        high_res = (high_res[0].to(self.device, non_blocking=True),
                    high_res[1].to(self.device, non_blocking=True))
        return img_feat, img_pe, high_res

    @torch.no_grad()
    def _encode_prompts(self, coords: torch.Tensor, labels: torch.Tensor):
        pe = getattr(self.model, "prompt_encoder", None) or getattr(self.model, "sam_prompt_encoder", None)
        if pe is None:
            raise AttributeError("No prompt_encoder / sam_prompt_encoder in model")
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

    @torch.no_grad()
    def _decode_mask(self, image_feat, image_pe, sparse_pe, dense_pe, high_res):
        md = getattr(self.model, "mask_decoder", None) or getattr(self.model, "sam_mask_decoder", None)
        if md is None:
            raise AttributeError("No mask_decoder / sam_mask_decoder in model")

        # 对齐 dense prompt 空间
        if dense_pe is not None and (dense_pe.shape[-2:] != image_feat.shape[-2:]):
            dense_pe = F.interpolate(dense_pe, size=image_feat.shape[-2:], mode="bilinear", align_corners=False)

        # high_res features：有的实现要求传入两个金字塔特征，且通道数要匹配
        feat_s0, feat_s1 = high_res
        tgt_s1, tgt_s0 = self._probe_dc_out_channels(md, image_feat)  # 注意顺序
        if feat_s1.shape[1] != tgt_s1: feat_s1 = self._match_channels(feat_s1, tgt_s1)
        if feat_s0.shape[1] != tgt_s0: feat_s0 = self._match_channels(feat_s0, tgt_s0)
        hr = (feat_s0, feat_s1)

        out = md(image_embeddings=image_feat,
                 image_pe=image_pe,
                 sparse_prompt_embeddings=sparse_pe,
                 dense_prompt_embeddings=dense_pe,
                 multimask_output=False,
                 repeat_image=True,
                 high_res_features=hr)
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict): return out.get("masks", out.get("mask_logits"))
        return out

    @staticmethod
    def _map_points_scale_xy(points: np.ndarray, sy: float, sx: float) -> torch.Tensor:
        if points is None or len(points) == 0:
            return torch.zeros((1,0,2), dtype=torch.float32)
        pts = np.asarray(points, dtype=np.float32).copy()
        pts[:, 0] *= sx; pts[:, 1] *= sy
        return torch.from_numpy(pts[:, :2]).unsqueeze(0).float()

    @torch.no_grad()
    def _pad_to(self, x: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
        """
        将 3D 张量 (C,H,W) 或 (1,H,W) 右侧/下侧零填充到 (C,Ht,Wt)。
        """
        assert x.ndim == 3, f"expect 3D tensor, got {x.shape}"
        _, H, W = x.shape
        if H == Ht and W == Wt:
            return x
        pad_h = max(0, Ht - H)
        pad_w = max(0, Wt - W)
        # F.pad 的顺序是 (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    @torch.no_grad()
    def encode_with_mask(self,
                        images_bgr: List[np.ndarray],
                        points_list: List[np.ndarray],
                        metas: Optional[List[dict]] = None,
                        conf_thr: float = 0.5):
        """
        返回：
        img_feats: (B,C,H*,W*) —— 每个样本特征 (C,H,W) 右/下零填充到 batch 最大 H/W
        masks    : (B,1,H*,W*) —— 同尺寸二值掩码
        """
        img_feats_raw, masks_raw = [], []
        Hs, Ws, Cs = [], [], []

        for i, (img_bgr, pts_np) in enumerate(zip(images_bgr, points_list)):
            # ---- 缓存 image 编码 ----
            key = None
            if metas is not None and i < len(metas):
                p = metas[i].get("image_path", None)
                if isinstance(p, str) and p:
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

            # ---- 点 -> 掩码（含兜底）----
            if pts_np is None or len(pts_np) == 0:
                mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
            else:
                coords = self._map_points_scale_xy(pts_np, sy, sx).to(self.device)
                labs = torch.from_numpy(
                    (np.asarray(pts_np, np.float32)[:, 2] > 0).astype(np.int64)
                ).unsqueeze(0).to(self.device)
                if labs.max() <= 0:
                    mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
                else:
                    sp, dp = self._encode_prompts(coords, labs)
                    mask_logits = self._decode_mask(img_feat, img_pe, sp, dp, high_res)
                    mask = torch.sigmoid(mask_logits)
                    if mask.shape[-2:] != img_feat.shape[-2:]:
                        mask = F.interpolate(mask, size=img_feat.shape[-2:], mode="bilinear", align_corners=False)
                    # 数值兜底
                    if (not torch.isfinite(mask).all()) or (mask.sum() <= 1e-5):
                        mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=self.device)
            # 二值化（对后续 masked GAP 更稳定）
            mask = (mask >= float(conf_thr)).float()

            feat3 = img_feat.squeeze(0).contiguous()  # (C,H,W)
            mask3 = mask.squeeze(0).contiguous()      # (1,H,W)

            img_feats_raw.append(feat3)
            masks_raw.append(mask3)
            Cs.append(int(feat3.shape[0]))
            Hs.append(int(feat3.shape[-2]))
            Ws.append(int(feat3.shape[-1]))

        # ---- 守卫：通道必须一致 ----
        if len(set(Cs)) != 1:
            raise RuntimeError(f"channel mismatch in batch: {set(Cs)}")

        # ---- 统一尺寸（右/下零填充到 batch 最大 H/W）----
        Ht = max(Hs) if Hs else 0
        Wt = max(Ws) if Ws else 0
        img_feats = [self._pad_to(f, Ht, Wt) for f in img_feats_raw]
        masks     = [self._pad_to(m, Ht, Wt) for m in masks_raw]

        # 再做一次强校验；若仍不一致，打印并抛错，方便定位
        shapes_f = [tuple(t.shape) for t in img_feats]
        shapes_m = [tuple(t.shape) for t in masks]
        if len({sh for sh in shapes_f}) != 1 or len({sh for sh in shapes_m}) != 1:
            print("[FATAL] after padding, feature shapes:", shapes_f)
            print("[FATAL] after padding, mask shapes   :", shapes_m)
            raise RuntimeError("Batch pad_to failed: shapes still mismatch.")

        return torch.stack(img_feats, dim=0), torch.stack(masks, dim=0)



    # ---------- 缓存工具 ----------
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


# ----------------------------- 掩码引导注意力池化 -----------------------------
class MaskedAttnPooler(nn.Module):
    """
    将 SAM2 的 (B,C,H,W) + 掩码 (B,1,H,W) 聚合为 (B, K*C_out)
    - K 个可学习 query（共享于所有样本）
    - Key/Value 由空间特征线性投影得到
    - 用掩码对注意力做 masked softmax（mask=0 的位置赋 -Inf）
    - 最终输出拼接 K 路的 pooled 向量（可选再拼上 masked GAP 特征，见 init 参数）
    """
    def __init__(self, in_dim: int, num_queries: int = 4, out_dim: Optional[int] = None,
                 add_masked_gap: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.k = int(num_queries)
        self.d = out_dim if out_dim is not None else in_dim
        self.add_masked_gap = bool(add_masked_gap)

        self.Wq = nn.Linear(self.in_dim, self.d, bias=False)
        self.Wk = nn.Linear(self.in_dim, self.d, bias=False)
        self.Wv = nn.Linear(self.in_dim, self.d, bias=False)

        # 可学习 query（K, in_dim）
        self.query = nn.Parameter(torch.randn(self.k, self.in_dim))
        nn.init.xavier_normal_(self.query)

        # 输出层（可选）：这里直接拼接，不降维
        self.out_dim = self.k * self.d + (self.in_dim if self.add_masked_gap else 0)

    def forward(self, feat_bchw: torch.Tensor, mask_b1hw: torch.Tensor) -> torch.Tensor:
        """
        feat_bchw: (B,C,H,W)
        mask_b1hw: (B,1,H,W)  (0/1)
        return:    (B, out_dim)
        """
        B, C, H, W = feat_bchw.shape
        HW = H * W
        x = feat_bchw.view(B, C, HW).permute(0, 2, 1)       # (B,HW,C)
        m = (mask_b1hw.view(B, 1, HW))                      # (B,1,HW) 0/1

        # 线性投影
        k = self.Wk(x)                                      # (B,HW,d)
        v = self.Wv(x)                                      # (B,HW,d)

        # queries：广播到 batch
        q = self.Wq(self.query).unsqueeze(0).expand(B, -1, -1)  # (B,K,d)

        # 注意力分数
        scores = torch.einsum("bkd,bhd->bkh", q, k) / math.sqrt(self.d)  # (B,K,HW)

        # 掩码：把 0 位置的分数置为 -Inf
        mask_inf = (1.0 - m).squeeze(1) * 1e9               # (B,HW)
        scores = scores - mask_inf.unsqueeze(1)             # (B,K,HW)
        attn = torch.softmax(scores, dim=-1)                # (B,K,HW)

        # 聚合
        pooled = torch.einsum("bkh,bhd->bkd", attn, v)      # (B,K,d)
        pooled_flat = pooled.reshape(B, -1)                 # (B, K*d)

        # 额外：masked GAP（与旧特征保持一定连续性）
        if self.add_masked_gap:
            denom = m.sum(dim=-1, keepdim=True).clamp_min(1.0)         # (B,1)
            gap = (x * m.transpose(1,2)).sum(dim=1) / denom.squeeze(1) # (B,C)
            out = torch.cat([pooled_flat, gap], dim=1)                  # (B, K*d + C)
        else:
            out = pooled_flat
        return out

# ----------------------------- 头部（CosFace / MLP / Linear） -----------------------------
class CosFace(nn.Module):
    """CosFace: s * (cos(theta) - m) on target logit"""
    def __init__(self, in_dim: int, n_classes: int, scale: float = 16.0, margin: float = 0.20):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_normal_(self.W)
        self.s = float(scale)
        self.m = float(margin)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,D), y: (B,)
        x_n = F.normalize(x, dim=1)                   # (B,D)
        w_n = F.normalize(self.W, dim=1)              # (C,D)
        cos = F.linear(x_n, w_n)                      # (B,C)
        if y is None:
            return self.s * cos
        # margin on target class
        onehot = F.one_hot(y, num_classes=self.W.size(0)).float()
        cos_m = cos - self.m * onehot
        return self.s * cos_m

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

# ----------------------------- 训练 / 验证 -----------------------------
def _apply_logit_adjust(logits: torch.Tensor, log_prior: Optional[torch.Tensor], tau: float):
    if (log_prior is None) or (tau is None) or (tau <= 0): return logits
    return logits - float(tau) * log_prior.view(1, -1).to(logits.device)

@torch.no_grad()
def _per_class_acc(logits: torch.Tensor, y: torch.Tensor, n_classes: int):
    preds = logits.argmax(dim=1)
    total = torch.zeros(n_classes, dtype=torch.long, device=logits.device)
    correct = torch.zeros(n_classes, dtype=torch.long, device=logits.device)
    for c in range(n_classes):
        mask = (y == c)
        total[c] = int(mask.sum().item())
        correct[c] = int((preds[mask] == c).sum().item()) if mask.any() else 0
    return total, correct

def train_one_epoch(extractor: Sam2OfficialWrapper,
                    pooler: MaskedAttnPooler,
                    head: nn.Module,
                    loader: DataLoader,
                    device: str,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    logger: TrainingLogger,
                    loss_fn, conf_thr: float,
                    max_grad_norm: float = 0.0,
                    log_prior: Optional[torch.Tensor] = None,
                    logit_adjust_tau: float = 0.0,
                    head_type: str = "cosface"):
    pooler.train(); head.train()
    running_loss, n = 0.0, 0

    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar, 1):
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)

        # 冻结 SAM2：仅前向
        with torch.no_grad():
            img_feat, mask = extractor.encode_with_mask(imgs, pts, metas, conf_thr=conf_thr)  # (B,C,H,W), (B,1,H,W)

        # 注意力池化 + 头
        feat = pooler(img_feat, mask)                    # (B, Dp)
        if head_type == "cosface":
            logits = head(feat, y=None)                  # margin 在 loss 前再加
            # 先做 logit-adjust，再在 loss 中替换 target logit
            logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)
            # 重算含 margin 的 target logit（不对其它类施加 margin）
            logits_m = head(feat, y=y)
            # 把 target 列替换到 logits 中（等价实现）
            logits = torch.where(F.one_hot(y, num_classes=logits.size(1)).bool(),
                                 logits_m, logits)
        else:
            logits = head(feat)
            logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)

        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(pooler.parameters()) + list(head.parameters()), max_grad_norm)
        optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        logger.write(f"epoch={epoch} step={step}/{len(loader)} train_loss={loss.item():.6f}")

    return running_loss / max(1, n)

@torch.inference_mode()
def evaluate(extractor: Sam2OfficialWrapper,
             pooler: MaskedAttnPooler,
             head: nn.Module,
             loader: DataLoader,
             device: str,
             epoch: int,
             logger: TrainingLogger,
             loss_fn, conf_thr: float,
             split_name: str,
             n_classes: int, id2name: Dict[int,str],
             log_prior: Optional[torch.Tensor] = None,
             logit_adjust_tau: float = 0.0,
             head_type: str = "cosface"):
    pooler.eval(); head.eval()
    total_loss, n = 0.0, 0
    correct = 0
    per_cls_total = torch.zeros(n_classes, dtype=torch.long)
    per_cls_correct = torch.zeros(n_classes, dtype=torch.long)

    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch} [{split_name}]", leave=False)
    for batch in pbar:
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)

        img_feat, mask = extractor.encode_with_mask(imgs, pts, metas, conf_thr=conf_thr)
        feat = pooler(img_feat, mask)

        if head_type == "cosface":
            logits = head(feat, y=None)
            logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)
            logits_m = head(feat, y=y)
            logits = torch.where(F.one_hot(y, num_classes=logits.size(1)).bool(),
                                 logits_m, logits)
        else:
            logits = head(feat)
            logits = _apply_logit_adjust(logits, log_prior, logit_adjust_tau)

        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()

        tot, cor = _per_class_acc(logits, y, n_classes)
        per_cls_total += tot.cpu()
        per_cls_correct += cor.cpu()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)
    logger.write(f"epoch={epoch} {split_name}_loss={avg_loss:.6f} {split_name}_acc={acc:.6f}")

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
    for p in str(s).split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(SMALLFILE_ROOT))
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--label-map", type=str, default=None)

    ap.add_argument("--sam2-cfg", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_l.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_large.pt"))

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out-dir", type=str, default=str(CKPT_ROOT))
    ap.add_argument("--resize", type=int, default=None)

    # Pooler / Head
    ap.add_argument("--num-queries", type=int, default=4, help="注意力池化的 query 数 K")
    ap.add_argument("--add-masked-gap", action="store_true", help="是否在输出中拼接 masked GAP 向量")
    ap.add_argument("--head", choices=["cosface","mlp","linear"], default="cosface")
    ap.add_argument("--hidden", type=str, default="0")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=16.0, help="CosFace scale s")
    ap.add_argument("--margin", type=float, default=0.20, help="CosFace margin m")

    # Balancing / loss / logit-adjust
    ap.add_argument("--balance", choices=["none","weights","sampler","auto"], default="auto")
    ap.add_argument("--reweight-alpha", type=float, default=0.5)
    ap.add_argument("--bg-factor", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--logit-adjust", type=float, default=0.5)  # 建议配合使用
    ap.add_argument("--focal", action="store_true", help="用 Focal Loss(gamma=2)，alpha 来自类权重")

    # 背景点策略 + 掩码阈值
    ap.add_argument("--bg-mask-mode", choices=["pos","global","mix"], default="mix")
    ap.add_argument("--conf-thr", type=float, default=0.5, help="mask 二值化阈值(注意力用)")

    # Resume（只恢复 pooler+head）
    ap.add_argument("--resume", type=str, default=None)

    args = ap.parse_args()
    if args.resize is not None:
        print("[WARN] Prefer --resize=None; wrapper 会对齐到 SAM2 合法尺寸。")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else (data_root / "manifest.csv")
    label_map_path = Path(args.label_map) if args.label_map else (data_root / "label_map.json")
    out_dir = Path(args.out_dir) if args.out_dir else CKPT_ROOT
    ensure_dir(out_dir); ensure_dir(SMALLFILE_ROOT)

    # Split or reuse
    train_csv = SMALLFILE_ROOT / "train_manifest.csv"
    val_csv   = SMALLFILE_ROOT / "val_manifest.csv"
    test_csv  = SMALLFILE_ROOT / "test_manifest.csv"
    if train_csv.exists() and val_csv.exists():
        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)
        df_test  = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame(columns=df_train.columns)
    else:
        df_all = pd.read_csv(manifest_path)
        df_train, df_val, df_test = _stable_train_val_test_split(df_all, seed=args.seed, train_ratio=0.8, val_ratio=0.1)
        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)
        df_test.to_csv(test_csv, index=False)

    lm = load_label_map(label_map_path)
    tool2id = lm["tool_to_id"]
    assert "background" in tool2id and tool2id["background"] == 0, "label_map.json must include background=0"
    id2name = {int(v): str(k) for k, v in tool2id.items()}
    n_classes = len(tool2id)

    # Datasets / Loaders
    ds_train = FramePointDataset(train_csv, label_map_path, resize=args.resize, bg_mask_mode=args.bg_mask_mode)
    ds_val   = FramePointDataset(val_csv,   label_map_path, resize=args.resize, bg_mask_mode=args.bg_mask_mode) if len(df_val)  else None
    ds_test  = FramePointDataset(test_csv,  label_map_path, resize=args.resize, bg_mask_mode=args.bg_mask_mode) if len(df_test) else None

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True) if ds_val else None
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          collate_fn=collate_varlen, pin_memory=True) if ds_test else None

    # Class stats / priors
    counts, train_dist, imb_ratio, priors = _class_stats(train_csv, label_map_path)
    if train_dist is not None:
        print(f"[CHECK] train distribution per class = {train_dist} | imbalance ratio={imb_ratio:.2f}")
    class_weights_ce = _ce_class_weights_from_counts(counts) if counts is not None else None
    log_prior = torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32, device=device)) if priors is not None else None

    # Sampler（可选）
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

    # Build frozen encoder
    extractor = Sam2OfficialWrapper(args.sam2_cfg, args.sam2_ckpt, device=device, cache_size=128)

    # Probe C
    probe = next(iter(dl_train))
    with torch.no_grad():
        img_feat_probe, mask_probe = extractor.encode_with_mask(probe["images"][:1], probe["points"][:1], probe["meta"][:1], conf_thr=args.conf_thr)
    in_dim = int(img_feat_probe.shape[1])  # C

    # Pooler + Head
    pooler = MaskedAttnPooler(in_dim=in_dim, num_queries=args.num_queries, add_masked_gap=args.add_masked_gap).to(device)
    head_type = args.head
    if head_type == "cosface":
        head = CosFace(pooler.out_dim, n_classes, scale=args.scale, margin=args.margin).to(device)
    elif head_type == "mlp":
        hidden_list = _parse_hidden_list(args.hidden)
        h = hidden_list[0] if hidden_list else 0
        head = MLPHead(pooler.out_dim, n_classes, hidden=h, drop=args.drop).to(device)
    else:  # linear
        head = MLPHead(pooler.out_dim, n_classes, hidden=0, drop=0.0).to(device)

    # Optim / Sched / Loss
    opt = torch.optim.AdamW(list(pooler.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.05)
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

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    # Resume
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        if "pooler" in ckpt and "head" in ckpt:
            pooler.load_state_dict(ckpt["pooler"], strict=False)
            head.load_state_dict(ckpt["head"], strict=False)
            print(f"[RESUME] Loaded pooler+head from {args.resume}")
        elif "head_state" in ckpt:
            head.load_state_dict(ckpt["head_state"], strict=False)
            print(f"[RESUME] Loaded head_state from {args.resume}")
        else:
            print(f"[RESUME] Unrecognized checkpoint keys in {args.resume}")

    # Logger
    log_file = SMALLFILE_ROOT / "train_attnproto_log.txt"
    logger = TrainingLogger(log_file)

    try:
        best_acc = -1.0
        best_epoch = -1
        patience_left = args.patience
        best_path = out_dir / "best_head_attnproto.pt"

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(
                extractor, pooler, head, dl_train, device, opt, epoch, logger,
                loss_fn=loss_fn, conf_thr=args.conf_thr,
                max_grad_norm=1.0, log_prior=log_prior, logit_adjust_tau=args.logit_adjust,
                head_type=head_type
            )

            va_loss, va_acc = evaluate(
                extractor, pooler, head, dl_val, device, epoch, logger,
                loss_fn=loss_fn, conf_thr=args.conf_thr, split_name="val",
                n_classes=n_classes, id2name=id2name,
                log_prior=log_prior, logit_adjust_tau=args.logit_adjust,
                head_type=head_type
            ) if dl_val else (0.0, 0.0)

            print(f"[{epoch:02d}] train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} val_acc {va_acc:.3f}")
            if sched is not None:
                sched.step()

            improved = (dl_val is None) or (va_acc > best_acc)
            if improved:
                best_acc = va_acc
                best_epoch = epoch
                patience_left = args.patience
                torch.save({
                    "pooler": pooler.state_dict(),
                    "head": head.state_dict(),
                    "in_dim": in_dim,
                    "pooler_out_dim": pooler.out_dim,
                    "n_classes": n_classes,
                    "tool_to_id": tool2id,
                    "args": vars(args),
                }, str(best_path))
                logger.write(f"epoch={epoch} SAVED best_attnproto -> {best_path}")
            else:
                patience_left -= 1

            if (epoch % 5 == 0) or (epoch == args.epochs):
                ep_path = out_dir / f"attnproto_epoch{epoch:03d}.pt"
                torch.save({
                    "pooler": pooler.state_dict(),
                    "head": head.state_dict(),
                    "in_dim": in_dim,
                    "pooler_out_dim": pooler.out_dim,
                    "n_classes": n_classes,
                    "tool_to_id": tool2id,
                    "args": vars(args),
                }, str(ep_path))
                logger.write(f"epoch={epoch} SAVED periodic_attnproto -> {ep_path}")

            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best val acc={best_acc:.4f} (epoch {best_epoch}).")
                logger.write(f"early_stop best_acc={best_acc:.6f} best_epoch={best_epoch}")
                break

        # Optional test
        if dl_test:
            te_loss, te_acc = evaluate(
                extractor, pooler, head, dl_test, device, epoch=best_epoch, logger=logger,
                loss_fn=loss_fn, conf_thr=args.conf_thr, split_name="test",
                n_classes=n_classes, id2name=id2name,
                log_prior=log_prior, logit_adjust_tau=args.logit_adjust,
                head_type=head_type
            )
            print(f"[TEST] loss {te_loss:.4f} acc {te_acc:.3f}")
            logger.write(f"test_loss={te_loss:.6f} test_acc={te_acc:.6f}")

        print(f"Done. Best val acc={best_acc:.4f} at epoch {best_epoch}. Saved to: {best_path}")
        logger.write(f"done best_acc={best_acc:.6f} best_epoch={best_epoch} path={best_path}")
    finally:
        logger.close()

if __name__ == "__main__":
    main()


# python /home/wcheng31/sam2_classify/train_sam2_classify_attnproto.py \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --data-root /home/wcheng31/sam2_classify/config \
#   --out-dir   /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier \
#   --epochs 20 --batch-size 128 --workers 4 \
#   --balance auto --reweight-alpha 0.5 --bg-factor 0.5 \
#   --bg-mask-mode mix --logit-adjust 0.5 \
#   --head cosface --scale 16 --margin 0.20 \
#   --num-queries 4 --conf-thr 0.5