#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 点提示分割 + 分类评测（最小增量版，带 Hydra 初始化）
- 掩码：用 backup 构建器 build_sam2_video_predictor
- 特征：SAM2 image_encoder 取最高分辨率 4D 特征 +（可选）位置编码
- 分类：加载训练好的 ViT 轻量头（best_head.pt），打印 overall/per-class acc
- 可视化：左原图 / 右遮罩+点+TopK

Author: WZC + ChatGPT (2025-09 -> ViT 头最小改动版)
"""

import os, json, argparse, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# ----------------- 常量路径（与你原来的保持一致） -----------------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"

# ----------------- Hydra 初始化：关键修复 -----------------
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

def setup_hydra_configs():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path="configs/sam2", version_base="1.2")

# ----------------- SAM2 构建（使用 backup 构建器） -----------------
from sam2.backup.build_sam import build_sam2, build_sam2_video_predictor

# =========================================================
# 数据集
# =========================================================
def pd_read_csv_fast(p: Path):
    return pd.read_csv(p)

def load_label_map(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

class FramePointDataset(Dataset):
    def __init__(self, manifest_csv: Path, label_map_json: Path):
        self.df = pd_read_csv_fast(manifest_csv)
        self.lm = load_label_map(label_map_json)
        self.tool2id = {str(k): int(v) for k, v in self.lm["tool_to_id"].items()}

        def has_points(s: str) -> bool:
            try:
                arr = json.loads(s) if isinstance(s, str) and s.strip() else []
                return len(arr) > 0
            except Exception:
                return False
        self.df = self.df[self.df["points_json"].apply(has_points)].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["image_path"])
        H, W = img.shape[:2]
        pts = json.loads(row["points_json"])
        pts_out = []
        for p in pts:
            x = float(np.clip(p[0], 0, W-1)); y = float(np.clip(p[1], 0, H-1))
            lab = 1.0 if float(p[2]) > 0 else 0.0
            pts_out.append([x, y, lab])
        tool = str(row["tool"])
        tool_id = int(self.tool2id[tool])
        return {
            "image": img,
            "points": np.asarray(pts_out, np.float32),
            "tool_id": tool_id,
            "meta": {"image_path": row["image_path"], "tool": tool}
        }

def choose_split_csv() -> Path:
    test_csv = SMALLFILE_ROOT / "test_manifest.csv"
    val_csv  = SMALLFILE_ROOT / "val_manifest.csv"
    mf_csv   = SMALLFILE_ROOT / "manifest.csv"
    if test_csv.exists(): return test_csv
    if val_csv.exists():  return val_csv
    return mf_csv

# =========================================================
# ViT 轻量头（与训练一致）
# =========================================================
class ViTTokenHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int,
                 num_layers: int = 2, num_heads: int = 4,
                 mlp_ratio: float = 4.0, p_drop: float = 0.05,
                 max_tokens: int = 1024):
        super().__init__()
        import math as _math
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
        # img_feat: [B,C,H,W], mask: [B,1,H,W] in [0,1], pos: [B,C,H,W] or None
        B,C,H,W = img_feat.shape
        N = H * W
        if N > self.max_tokens:
            s = (N / float(self.max_tokens)) ** 0.5
            Ht = max(1, int(H / s + 0.5))
            Wt = max(1, int(W / s + 0.5))
            while Ht * Wt > self.max_tokens:
                if Ht >= Wt and Ht > 1: Ht -= 1
                elif Wt > 1:            Wt -= 1
                else:                   break
            img_feat = F.adaptive_avg_pool2d(img_feat, (Ht, Wt))
            if pos is not None:  pos  = F.adaptive_avg_pool2d(pos,  (Ht, Wt))
            if mask is not None: mask = F.adaptive_max_pool2d(mask, (Ht, Wt))
            H, W = Ht, Wt

        x = img_feat if pos is None else (img_feat + pos)
        x = x.permute(0,2,3,1).reshape(B, H*W, C)

        key_padding = None
        if mask is not None:
            thr = 0.3
            keep = (mask > thr).flatten(1)          # [B,N]
            Kmin = min(64, H*W)
            for i in range(B):
                if int(keep[i].sum().item()) < Kmin:
                    vals = mask[i,0].flatten()
                    k = min(Kmin, vals.numel())
                    topk = torch.topk(vals, k=k, dim=0).indices
                    keep[i].zero_(); keep[i, topk] = True
            key_padding = (~keep).bool()            # True 表示 pad

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # [B,1+N,C]
        if key_padding is not None:
            pad0 = torch.zeros(B,1, dtype=torch.bool, device=x.device)
            key_padding = torch.cat([pad0, key_padding], dim=1)

        x = self.enc(x, src_key_padding_mask=key_padding)
        cls_out = self.norm(x[:,0])
        return self.fc(cls_out)

# =========================================================
# SAM2 编码器（返回 4D 特征 +（可选）匹配位置编码）
# =========================================================
class Sam2ImageEncoder(nn.Module):
    def __init__(self, cfg: str, ckpt: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = build_sam2(cfg, ckpt, device=device)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        pm = getattr(self.model.image_encoder, "pixel_mean", [123.675, 116.28, 103.53])
        ps = getattr(self.model.image_encoder, "pixel_std",  [58.395, 57.12, 57.375])
        pm = torch.as_tensor(pm, dtype=torch.float32).view(1,3,1,1) / 255.0
        ps = torch.as_tensor(ps, dtype=torch.float32).view(1,3,1,1) / 255.0
        self.register_buffer("pixel_mean", pm)
        self.register_buffer("pixel_std",  ps)

    def _legal_hw_from_orig(self, H0: int, W0: int):
        enc = getattr(self.model, "image_encoder", None)
        trunk = getattr(enc, "trunk", None)
        conv = trunk.patch_embed.proj
        k_h, k_w = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        s_h, s_w = conv.stride      if isinstance(conv.stride, tuple)      else (conv.stride, conv.stride)
        p_h, p_w = conv.padding     if isinstance(conv.padding, tuple)     else (conv.padding, conv.padding)
        win = getattr(trunk, "window_size", 16)
        if isinstance(win, (list, tuple)): win = int(win[0])

        def tokens(n, k, s, p): return int((n + 2 * p - k) // s + 1)
        def size(n_tok, k, s, p): return int(s * (n_tok - 1) + k - 2 * p)
        import math
        t_h0 = max(1, tokens(H0, k_h, s_h, p_h))
        t_w0 = max(1, tokens(W0, k_w, s_w, p_w))
        t_h = max(win, int(math.ceil(t_h0 / win)) * win)
        t_w = max(win, int(math.ceil(t_w0 / win)) * win)
        H_in = size(t_h, k_h, s_h, p_h); W_in = size(t_w, k_w, s_w, p_w)
        sy = H_in / max(1.0, float(H0)); sx = W_in / max(1.0, float(W0))
        return H_in, W_in, sy, sx

    @torch.no_grad()
    def encode(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """返回: (img_feat[B=1,C,Hf,Wf], pos 或 None)"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]
        H_in, W_in, _, _ = self._legal_hw_from_orig(H0, W0)
        if (H_in, W_in) != (H0, W0):
            img_rgb = cv2.resize(img_rgb, (W_in, H_in), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(self.device, non_blocking=True)
        x = (x - self.pixel_mean) / self.pixel_std

        out = self.model.image_encoder(x)
        # 尝试按训练脚本方式拿到最大分辨率的 4D feat 和匹配 pos_enc
        img_feat, img_pe = None, None
        if isinstance(out, dict) and ("vision_features" in out):
            vfeats = out["vision_features"]
            vpos   = out.get("vision_pos_enc", None)
            if isinstance(vfeats, torch.Tensor): vfeats = [vfeats]
            if isinstance(vpos,   torch.Tensor): vpos   = [vpos]
            vpos_list = list(vpos) if isinstance(vpos, (list, tuple)) else []
            cand = [(i,t) for i,t in enumerate(vfeats) if torch.is_tensor(t) and t.ndim==4]
            if cand:
                _, img_feat = max(cand, key=lambda x: int(x[1].shape[-2])*int(x[1].shape[-1]))
                Hf,Wf = int(img_feat.shape[-2]), int(img_feat.shape[-1])
                for p in vpos_list:
                    if torch.is_tensor(p) and p.ndim>=3 and int(p.shape[-2])==Hf and int(p.shape[-1])==Wf:
                        img_pe = p; break
        if img_feat is None:
            feats = []
            def collect(o):
                if torch.is_tensor(o) and o.ndim==4: feats.append(o)
                elif isinstance(o, dict):
                    for v in o.values(): collect(v)
                elif isinstance(o, (list,tuple)):
                    for v in o: collect(v)
            collect(out)
            if not feats:
                raise RuntimeError("image_encoder returned no 4D features")
            img_feat = max(feats, key=lambda t: int(t.shape[-2])*int(t.shape[-1]))
        return img_feat.to(self.device), (img_pe.to(self.device) if isinstance(img_pe, torch.Tensor) else None)

# =========================================================
# 掩码（predictor） → ViT 所需的 mask tensor
# =========================================================
@torch.no_grad()
def predictor_mask_from_points(predictor, img_bgr: np.ndarray, pts_np: np.ndarray, mask_thr: float = 0.5) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    tmpdir = tempfile.mkdtemp(prefix="sam2_one_")
    try:
        fpath = os.path.join(tmpdir, "0000000.jpg")
        cv2.imwrite(fpath, img_bgr)
        state = predictor.init_state(video_path=tmpdir); predictor.reset_state(state)
        if pts_np.size > 0:
            xy = pts_np[:, :2].astype(np.float32)
            lab = (pts_np[:, 2] > 0).astype(np.int64)
            if lab.sum() == 0: lab[0] = 1
            oid = 1
            if hasattr(predictor, "add_new_points"):
                predictor.add_new_points(state, 0, oid, xy, lab)
            else:
                predictor.add_new_points_or_box(state, 0, oid, xy, lab)
        raw = None
        for _, obj_ids, logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(obj_ids):
                raw = (torch.sigmoid(logits[i]) > mask_thr).float().cpu().numpy()
        if raw is None:
            return np.zeros((H,W), np.uint8)
        m = raw[0] if raw.ndim==3 else raw
        if m.shape != (H,W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return (m > 0.5).astype(np.uint8)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def mask_hw_to_feat_mask(mask_hw: np.ndarray, feat_4d: torch.Tensor) -> torch.Tensor:
    """mask_hw[H,W] -> [1,1,Hf,Wf] float"""
    _, C, Hf, Wf = feat_4d.shape
    m = mask_hw.astype(np.float32)
    if m.shape != (Hf, Wf):
        m = cv2.resize(m, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
    m_t = torch.from_numpy(m).view(1,1,Hf,Wf).to(feat_4d.device)
    return m_t

# =========================================================
# 评测 & 可视化（保持原输出/统计风格）
# =========================================================
@torch.no_grad()
def evaluate_and_visualize(ds: Dataset,
                           predictor,
                           encoder: Sam2ImageEncoder,
                           head: ViTTokenHead,
                           id2name: Dict[int,str],
                           out_dir: Path,
                           num_vis: int = 50,
                           mask_thr: float = 0.5,
                           topk: int = 3,
                           device: str = "cuda"):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(id2name)
    total = 0; correct = 0
    per_total = [0]*n_classes; per_corr = [0]*n_classes

    # 可视化索引：按 image_path 去重并采样
    if hasattr(ds, "df") and "image_path" in ds.df.columns:
        dfv = ds.df.drop_duplicates(subset=["image_path"]).sample(frac=1.0, random_state=42)
        vis_idx = dfv.index.tolist()[:num_vis]
    else:
        vis_idx = list(range(min(num_vis, len(ds))))

    pbar = tqdm(range(len(ds)), ncols=100, desc="[eval]")
    vis_written = 0
    for i in pbar:
        sample = ds[i]
        img = sample["image"]; pts = sample["points"]; gt = int(sample["tool_id"])

        # 掩码（用于可视化 + 作为 ViT 的 token 选择）
        mask_hw = predictor_mask_from_points(predictor, img, pts, mask_thr=mask_thr)

        # 特征（4D） + 可选 pos
        feat4, pos = encoder.encode(img)                # [1,C,Hf,Wf], [1,C,Hf,Wf] or None
        mask_t = mask_hw_to_feat_mask(mask_hw, feat4)   # [1,1,Hf,Wf]
        logits = head(feat4.to(device), mask_t.to(device), (pos.to(device) if pos is not None else None))
        pred   = int(logits.argmax(dim=1).item())

        # 统计
        total += 1; correct += int(pred == gt)
        per_total[gt] += 1; per_corr[gt] += int(pred == gt)
        pbar.set_postfix(acc=f"{correct/max(1,total):.3f}")

        # 可视化
        if vis_written < num_vis and i in vis_idx:
            prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            top = prob.argsort()[::-1][:topk]
            right = img.copy()
            # 将掩码区域暗化以便对比（与原逻辑一致的视觉效果）
            right[mask_hw.astype(bool)] = 0
            for (x,y,lab) in pts:
                c = (0,255,0) if lab>0 else (0,0,255)
                cv2.circle(right, (int(round(x)), int(round(y))), 5, c, -1, cv2.LINE_AA)
                cv2.circle(right, (int(round(x)), int(round(y))), 6, (0,0,0), 1, cv2.LINE_AA)
            y0 = 28
            cv2.putText(right, f"GT: {id2name.get(gt,str(gt))} (p={prob[gt]:.3f})",
                        (8,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            y = y0 + 28
            for r,k in enumerate(top,1):
                cv2.putText(right, f"Top{r}: {id2name.get(int(k),str(k))} {prob[k]:.3f}",
                            (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                y += 24
            canvas = np.concatenate([img, right], axis=1)
            cv2.imwrite(str(out_dir / f"vis_{vis_written:04d}.jpg"), canvas)
            vis_written += 1

    # 打印指标（保持你原来的风格）
    overall = correct / max(1,total)
    print(f"\n=== Overall ===\nAcc: {overall:.4f}  (#samples={total})")
    print("\n=== Per-class Acc ===")
    for c in range(n_classes):
        n = per_total[c]
        acc_c = per_corr[c] / max(1,n)
        print(f"{c:2d} {id2name.get(c,str(c)):>18s}: acc={acc_c:.4f}  (n={n})")

# =========================================================
# Main（CLI 保持不变；只把 head 替换为 ViT）
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", choices=["prepared"], default="prepared")
    ap.add_argument("--sam2-cfg",  type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_l.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_large.pt"))
    ap.add_argument("--ckpt", type=str, required=True, help="head ckpt (best_head.pt)")
    ap.add_argument("--num-vis", type=int, default=50)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out-dir", type=str, default=str(CKPT_ROOT / "vis_test"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据
    label_map_path = SMALLFILE_ROOT / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"missing label_map: {label_map_path}")
    label_map = load_label_map(label_map_path)
    tool2id = {str(k): int(v) for k,v in label_map["tool_to_id"].items()}
    id2name = {int(v): str(k) for k,v in tool2id.items()}

    csv_path = choose_split_csv()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    ds = FramePointDataset(csv_path, label_map_path)

    # 关键：先初始化 Hydra，再 build predictor
    setup_hydra_configs()
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device)
    print("SAM-2 predictor ready.")

    # 编码器
    encoder = Sam2ImageEncoder(args.sam2_cfg, args.sam2_ckpt, device=device)

    # ==== 从 ckpt 恢复 ViT 轻量头（按训练脚本参数） ====
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ck_args = ckpt.get("args", {}) or {}
    # 通过探针拿 in_dim（C）
    probe_img = ds[0]["image"]
    probe_feat, _ = encoder.encode(probe_img)
    in_dim  = int(probe_feat.shape[1])
    n_cls   = int(ckpt.get("n_classes", len(tool2id)))
    if n_cls != len(tool2id):
        print(f"[WARN] n_classes mismatch: ckpt={n_cls}, current={len(tool2id)} (use current).")
        n_cls = len(tool2id)

    vit_layers = int(ck_args.get("vit_layers", 2))
    vit_heads  = int(ck_args.get("vit_heads", 4))
    vit_drop   = float(ck_args.get("vit_drop", 0.05))
    vit_mlp    = float(ck_args.get("vit_mlp_ratio", ck_args.get("vit_mlpp_ratio", 4.0)))
    vit_max_tk = int(ck_args.get("vit_max_tokens", 1024))

    head = ViTTokenHead(in_dim=in_dim, n_classes=n_cls,
                        num_layers=vit_layers, num_heads=vit_heads,
                        mlp_ratio=vit_mlp, p_drop=vit_drop,
                        max_tokens=vit_max_tk).to(device)
    state = ckpt.get("head_state", ckpt)
    head.load_state_dict(state, strict=False)  # 非严格容错
    head.eval()

    # 评测与可视化（保持你原来脚本的输出风格）
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_and_visualize(ds, predictor, encoder, head, id2name,
                           out_dir=out_dir, num_vis=args.num_vis,
                           mask_thr=args.mask_thr, topk=args.topk, device=device)

if __name__ == "__main__":
    main()

# python /home/wcheng31/sam2_classify/test_sam2_classify_vit.py \
#   --eval-set prepared \
#   --sam2-cfg /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/vit_head/best_head.pt \
#   --num-vis 50 --mask-thr 0.5 --topk 3 \
#   --out-dir /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/vis_test_min
