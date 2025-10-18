#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse, math, time, hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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

# ---- plotting ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ====== hydra / sam2 build ======
if "/home/wcheng31/sam2" not in sys.path:
    sys.path.append("/home/wcheng31/sam2")
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

def _setup_hydra_configs():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path="configs/sam2", version_base="1.2")

try:
    from sam2.backup.build_sam import build_sam2
except Exception:
    from sam2.build_sam import build_sam2

# ====== constants / paths ======
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")

# =============== Dataset ===============
class FramePointDataset(Dataset):
    """
    兼容三类清单：
    - combined/cholect: 有 'tool' 列和 'points_json'
    - ood5: 有 'mapped_class' 或 'class_id'，同样有 'points_json'
    """
    def __init__(self, manifest_csv: Path, label_map_json: Path, resize: Optional[int] = None):
        super().__init__()
        self.df = pd.read_csv(manifest_csv)
        with open(label_map_json, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.tool2id = self.label_map["tool_to_id"]
        self.resize = resize

        # 统一出 "tool" 列（优先已有的 tool；否则用 mapped_class；否则 class_id->name）
        if "tool" not in self.df.columns:
            if "mapped_class" in self.df.columns:
                self.df["tool"] = self.df["mapped_class"].astype(str)
            elif "class_id" in self.df.columns:
                inv = {int(v): k for k, v in self.tool2id.items()}
                self.df["tool"] = [inv.get(int(cid), "background") for cid in self.df["class_id"].tolist()]
            else:
                raise RuntimeError("manifest 缺少 tool / mapped_class / class_id 任一字段")
        # 过滤掉没有 points_json 或为空的
        def _has_pos_points(s: str) -> bool:
            try:
                arr = json.loads(s) if isinstance(s, str) and s.strip() else []
                return len(arr) > 0
            except Exception:
                return False
        if "points_json" in self.df.columns:
            self.df = self.df[self.df["points_json"].apply(_has_pos_points)].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def _load_img(self, p: str):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        if self.resize and self.resize > 0:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_img(row["image_path"])
        H, W = img.shape[:2]
        pts_raw = json.loads(row["points_json"]) if isinstance(row["points_json"], str) and row["points_json"].strip() else []
        pts = []
        for p in pts_raw:
            if len(p) < 2: continue
            x = float(np.clip(p[0], 0, W-1)); y = float(np.clip(p[1], 0, H-1))
            label = 1.0 if (len(p) >= 3 and float(p[2]) > 0) else 1.0 # 测试阶段统一当正点
            pts.append([x, y, label])

        tool = str(row["tool"])
        tool_id = self.tool2id.get(tool, None)
        if tool_id is None:
            # 退路：严格字符串匹配
            for k,v in self.tool2id.items():
                if str(k) == tool: tool_id = v; break
        if tool_id is None:
            raise KeyError(f"Tool '{tool}' 不在 label_map 中")

        return {
            "image": img,
            "points": np.asarray(pts, dtype=np.float32) if pts else np.zeros((0,3), np.float32),
            "target": int(tool_id),
            "meta": {
                "image_path": row["image_path"],
                "tool": tool,
                "H": H, "W": W,
            }
        }

def collate_varlen(batch):
    return {
        "images":  [b["image"]  for b in batch],
        "points":  [b["points"] for b in batch],
        "targets": torch.tensor([b["target"] for b in batch], dtype=torch.long),
        "meta":    [b["meta"]   for b in batch]
    }

# =============== SAM2 Wrapper (统一推理) ===============
class Sam2Wrapper(nn.Module):
    """
    单张图：
      - 手动 resize -> 送入 image_encoder
      - prompt_encoder(points) -> mask_decoder -> mask
      - mask 加权池化 image_feat -> 向量
    也支持返回 4D（ViTTokenHead 用）
    """
    def __init__(self, cfg: str, ckpt: str, device: str = "cuda", max_edge: Optional[int] = None):
        super().__init__()
        self.device = device
        self.max_edge = max_edge
        _setup_hydra_configs()
        # 允许直接传文件名（相对 sam2/configs/sam2）
        cfg = os.path.basename(cfg) if cfg.endswith(".yaml") else cfg
        self.model = build_sam2(cfg, ckpt, device=device)
        self._norm_cached = None

    # ---- norm ----
    def _get_norm(self):
        if self._norm_cached is not None: return self._norm_cached
        for obj in [self.model, getattr(self.model, "image_encoder", None)]:
            pm = getattr(obj, "pixel_mean", None); ps = getattr(obj, "pixel_std", None)
            if pm is not None and ps is not None:
                pm = torch.as_tensor(pm, dtype=torch.float32).view(1,3,1,1)
                ps = torch.as_tensor(ps, dtype=torch.float32).view(1,3,1,1)
                if pm.max() > 1.5 or ps.max() > 1.5: pm = pm/255.0; ps = ps/255.0
                self._norm_cached = (pm.to(self.device), ps.to(self.device)); return self._norm_cached
        pm = torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1)
        ps = torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1)
        self._norm_cached = (pm, ps); return self._norm_cached

    def _prep(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]
        if self.max_edge is not None and max(H0, W0) > self.max_edge:
            s = self.max_edge / max(H0, W0)
            H1 = int(round(H0 * s)); W1 = int(round(W0 * s))
            img_rgb = cv2.resize(img_rgb, (W1, H1), interpolation=cv2.INTER_AREA)
        else:
            H1, W1 = H0, W0
        t = torch.from_numpy(img_rgb).permute(2,0,1).float().unsqueeze(0)/255.0
        pm, ps = self._get_norm()
        t = (t.to(self.device) - pm) / ps
        return t, (H0, W0), (H1, W1), (H1/H0, W1/W0)

    def _encode_prompts(self, coords: torch.Tensor, labels: torch.Tensor):
        pe = getattr(self.model, "prompt_encoder", None) or getattr(self.model, "sam_prompt_encoder", None)
        out = pe(points=(coords, labels), boxes=None, masks=None)
        if isinstance(out, (list, tuple)):
            return out[0], out[1]
        if isinstance(out, dict):
            return out.get("sparse_prompt_embeddings"), out.get("dense_prompt_embeddings")
        raise RuntimeError("unexpected prompt encoder output")

    def _decode_mask(self, image_feat, image_pe, sparse_pe, dense_pe, high_res):
        md = getattr(self.model, "mask_decoder", None) or getattr(self.model, "sam_mask_decoder", None)
        kwargs = dict(image_embeddings=image_feat, image_pe=image_pe,
                      sparse_prompt_embeddings=sparse_pe, dense_prompt_embeddings=dense_pe,
                      multimask_output=False, repeat_image=True)
        if high_res is not None:
            kwargs["high_res_features"] = high_res
        out = md(**kwargs)
        if isinstance(out, (list, tuple)): return out[0]
        if isinstance(out, dict): return out.get("masks", out.get("mask_logits"))
        return out

    def _get_image_embed(self, img_t: torch.Tensor):
        out = self.model.image_encoder(img_t)
        if isinstance(out, dict) and ("vision_features" in out):
            vfeats = out["vision_features"]
            vpos   = out.get("vision_pos_enc", None)
            levels = out.get("backbone_fpn", None)
            if isinstance(vfeats, torch.Tensor): vfeats = [vfeats]
            cand = [t for t in vfeats if torch.is_tensor(t) and t.ndim==4]
            img_feat = max(cand, key=lambda t: int(t.shape[-2])*int(t.shape[-1]))
            Hf,Wf = img_feat.shape[-2:]
            img_pe=None
            if isinstance(vpos, (list,tuple)):
                for p in vpos:
                    if torch.is_tensor(p) and p.ndim>=3 and int(p.shape[-2])==Hf and int(p.shape[-1])==Wf:
                        img_pe = p; break
            elif torch.is_tensor(vpos) and int(vpos.shape[-2])==Hf and int(vpos.shape[-1])==Wf:
                img_pe = vpos
            if isinstance(levels,(list,tuple)):
                levels = [x.to(img_feat.device) for x in levels if torch.is_tensor(x)]
            elif torch.is_tensor(levels):
                levels = [levels.to(img_feat.device)]
            else:
                levels = []
            if len(levels)>=2:
                levels_sorted = sorted(levels, key=lambda t:int(t.shape[-2])*int(t.shape[-1]), reverse=True)
                high_res = (levels_sorted[0], levels_sorted[1])
            elif len(levels)==1:
                high_res = (levels[0], levels[0])
            else:
                high_res = None
            return img_feat, (img_pe.to(img_feat.device) if isinstance(img_pe, torch.Tensor) else None), high_res
        # fallback
        tensors=[]
        def collect(o):
            if torch.is_tensor(o): tensors.append(o)
            elif isinstance(o, dict):
                for v in o.values(): collect(v)
            elif isinstance(o,(list,tuple)):
                for v in o: collect(v)
        collect(out)
        cand = [t for t in tensors if t.ndim==4]
        img_feat = max(cand, key=lambda t: int(t.shape[-2])*int(t.shape[-1]))
        return img_feat, None, None

    @staticmethod
    def _points_to_coords(pts: np.ndarray, sy: float, sx: float):
        if pts is None or len(pts)==0:
            return torch.zeros((1,0,2), dtype=torch.float32)
        arr = np.asarray(pts, dtype=np.float32).copy()
        arr[:,0] *= sx; arr[:,1] *= sy
        return torch.from_numpy(arr[:,:2]).unsqueeze(0).float()

    @torch.no_grad()
    def forward(self, images_bgr: List[np.ndarray], points_list: List[np.ndarray],
                metas: Optional[List[dict]]=None, return_4d: bool=False):
        feats_triplets = []
        for img_bgr, pts in zip(images_bgr, points_list):
            img_t, (H0,W0), (H1,W1), (sy,sx) = self._prep(img_bgr)
            img_feat, img_pe, high_res = self._get_image_embed(img_t)
            if (pts is None) or (len(pts)==0):
                mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
            else:
                coords = self._points_to_coords(pts, sy, sx).to(img_feat.device)
                labels = torch.ones((1, coords.shape[1]), dtype=torch.long, device=img_feat.device)
                sp, dp = self._encode_prompts(coords, labels)
                mask_logits = self._decode_mask(img_feat, img_pe, sp, dp, high_res)
                if mask_logits.shape[-2:] != img_feat.shape[-2:]:
                    mask_logits = F.interpolate(mask_logits, size=img_feat.shape[-2:], mode="bilinear", align_corners=False)
                mask = torch.sigmoid(mask_logits)
            feats_triplets.append((img_feat, mask, img_pe))
        if return_4d:
            # 对齐到最小空间尺寸
            sizes = [t[0].shape[-2:] for t in feats_triplets]
            th = min(h for h,_ in sizes); tw = min(w for _,w in sizes)
            def pool_to(x, size): 
                if x is None: return None
                return x if x.shape[-2:]==size else F.adaptive_avg_pool2d(x, size)
            img_b = torch.cat([pool_to(f, (th,tw)) for (f,_,_) in feats_triplets], dim=0)
            msk_b = torch.cat([pool_to(m, (th,tw)) for (_,m,_) in feats_triplets], dim=0)
            pe_b  = []
            for (_,_,pe) in feats_triplets:
                pe_b.append(torch.zeros_like(img_b[:1]) if pe is None else pool_to(pe, (th,tw)))
            pe_b = torch.cat(pe_b, dim=0)
            return img_b, msk_b, pe_b
        # pool -> vector
        vecs=[]
        for (f, m, _) in feats_triplets:
            v = (f*m).flatten(2).sum(dim=-1)/(m.flatten(2).sum(dim=-1)+1e-6)
            vecs.append(v.squeeze(0))
        return torch.stack(vecs, dim=0)

# =============== Heads ===============
class MLPHead(nn.Module):
    def __init__(self, in_dim:int, n_classes:int, hidden:int=0, drop:float=0.0):
        super().__init__()
        if hidden and hidden>0:
            self.fc1 = nn.Linear(in_dim, hidden)
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop) if drop>0 else nn.Identity()
            self.fc2 = nn.Linear(hidden, n_classes)
            self._deep=True
        else:
            self.fc = nn.Linear(in_dim, n_classes)
            self._deep=False
    def forward(self,x):
        if not self._deep: return self.fc(x)
        return self.fc2(self.drop(self.act(self.fc1(x))))

class CosineClassifier(nn.Module):
    def __init__(self, in_dim:int, n_classes:int, scale:float=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = float(scale)
    def forward(self,x):
        x_n = F.normalize(x, dim=1); w_n = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x_n, w_n)

class ViTTokenHead(nn.Module):
    def __init__(self, in_dim:int, n_classes:int,
                 num_layers:int=2, num_heads:int=4,
                 mlp_ratio:float=4.0, p_drop:float=0.05,
                 max_tokens:int=1024, min_keep_tokens:int=256):
        super().__init__()
        self.max_tokens=int(max_tokens); self.min_keep_tokens=int(min_keep_tokens)
        self.cls = nn.Parameter(torch.zeros(1,1,in_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads,
            dim_feedforward=int(mlp_ratio*in_dim), dropout=p_drop,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(in_dim)
        self.fc   = nn.Linear(in_dim, n_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def _pos(self, B,C,H,W, device):
        y = torch.arange(H, device=device).float(); x = torch.arange(W, device=device).float()
        yy,xx = torch.meshgrid(y,x, indexing="ij"); yy/=max(1.0,H); xx/=max(1.0,W)
        freqs=[1.0,2.0,4.0,8.0]
        pe=[torch.sin(2*math.pi*f*yy) for f in freqs]+[torch.cos(2*math.pi*f*yy) for f in freqs] + \
           [torch.sin(2*math.pi*f*xx) for f in freqs]+[torch.cos(2*math.pi*f*xx) for f in freqs]
        pe=torch.stack(pe, dim=-1)
        if pe.shape[-1]<C: pe=torch.cat([pe, torch.zeros(H,W,C-pe.shape[-1], device=device)], dim=-1)
        elif pe.shape[-1]>C: pe=pe[...,:C]
        return pe.permute(2,0,1).unsqueeze(0).expand(B,-1,-1,-1).contiguous()

    def forward(self, img_feat:torch.Tensor, mask:Optional[torch.Tensor]=None, pos:Optional[torch.Tensor]=None):
        B,C,H,W = img_feat.shape
        N = H*W
        if N>self.max_tokens:
            s=math.sqrt(N/float(self.max_tokens))
            Ht=max(1,int(H/s+0.5)); Wt=max(1,int(W/s+0.5))
            while Ht*Wt>self.max_tokens:
                if Ht>=Wt and Ht>1: Ht-=1
                elif Wt>1: Wt-=1
                else: break
            img_feat = F.adaptive_avg_pool2d(img_feat, (Ht,Wt))
            if pos  is not None: pos  = F.adaptive_avg_pool2d(pos, (Ht,Wt))
            if mask is not None: mask = F.adaptive_max_pool2d(mask,(Ht,Wt))
            H,W = Ht,Wt
        if pos is None:
            pos = self._pos(B,C,H,W,img_feat.device)
        x = (img_feat+pos).permute(0,2,3,1).reshape(B,H*W,C)
        key_padding=None
        if mask is not None:
            thr=0.3
            keep=(mask>thr).flatten(1)
            Kmin=min(self.min_keep_tokens, H*W)
            for i in range(B):
                if int(keep[i].sum())<Kmin:
                    vals=mask[i,0].flatten()
                    k=min(Kmin, vals.numel())
                    topk=torch.topk(vals,k=k,dim=0).indices
                    keep[i].zero_(); keep[i,topk]=True
            key_padding=(~keep).bool()
        cls=self.cls.expand(B,-1,-1)
        x=torch.cat([cls,x], dim=1)
        if key_padding is not None:
            pad0=torch.zeros(B,1, dtype=torch.bool, device=x.device)
            key_padding=torch.cat([pad0, key_padding], dim=1)
        x=self.enc(x, src_key_padding_mask=key_padding)
        cls_out=self.norm(x[:,0])
        return self.fc(cls_out)

# =============== metrics / viz ===============
@torch.no_grad()
def _compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int):
    conf = torch.zeros((n_classes, n_classes), dtype=torch.long, device=y_true.device)
    for t,p in zip(y_true, y_pred): conf[t,p]+=1
    eps=1e-12
    tp=conf.diag().float(); pp=conf.sum(0).float(); tpfn=conf.sum(1).float()
    prec = tp/(pp+eps); rec = tp/(tpfn+eps)
    f1 = 2*prec*rec/(prec+rec+eps)
    return {
        "acc": (y_true==y_pred).float().mean().item(),
        "macro_f1": torch.nanmean(f1).item(),
        "balanced_acc": torch.nanmean(rec).item(),
        "confusion": conf.cpu()
    }

def _save_confmat(cm: np.ndarray, id2name: Dict[int,str], path: Path, title: str):
    if not HAS_MPL:
        print("[WARN] matplotlib 不可用，跳过保存混淆矩阵")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        row = cm.sum(1, keepdims=True); cmn = np.divide(cm, row, out=np.zeros_like(cm, float), where=row>0)
    fig, ax = plt.subplots(figsize=(7,6), dpi=160)
    im=ax.imshow(cmn, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes=[id2name.get(i,str(i)) for i in range(cm.shape[0])]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, xlabel="Pred", ylabel="GT", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i, str(int(cm[i,j])), ha="center", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(str(path), bbox_inches="tight"); plt.close(fig)

# =============== ckpt loader / model builder ===============
def _build_and_load(args, tool2id: Dict[str,int], device: str):
    """
    返回 (extractor, head, head_type)
    自动识别 5 种：
      - 冻结 + 线性/MLP/Cosine（仅 head_state）
      - E2E ViTTokenHead（best_e2e.pt）
      - Finetune + (Cosine/MLP/Linear) 蒸馏（best_full_finetune.pt）
    """
    ckpt = torch.load(args.ckpt, map_location="cpu")
    has_sam2 = ("sam2_state" in ckpt) or any(k.startswith("model.") for k in ckpt.keys())
    has_head = ("head_state" in ckpt) or any(k.startswith("fc") or k.startswith("mlp") or k.startswith("weight") for k in ckpt.keys())

    # 构建 extractor
    extractor = Sam2Wrapper(args.sam2_cfg, args.sam2_ckpt, device=device, max_edge=args.max_input_edge).to(device)
    extractor.eval()

    # 自动推断 head 类型
    saved_args = ckpt.get("args", {})
    head_hint = saved_args.get("head", None)
    n_classes = ckpt.get("n_classes", len(tool2id))
    id2name = {int(v):k for k,v in tool2id.items()}

    def _mk_head_linear(in_dim): return MLPHead(in_dim, n_classes, hidden=0).to(device)
    def _mk_head_mlp(in_dim, hidden=0): return MLPHead(in_dim, n_classes, hidden=hidden or saved_args.get("hidden",0), drop=saved_args.get("drop",0.0)).to(device)
    def _mk_head_cos(in_dim): return CosineClassifier(in_dim, n_classes, scale=float(saved_args.get("scale",16.0))).to(device)
    def _mk_head_vit(in_dim):
        return ViTTokenHead(
            in_dim, n_classes,
            num_layers=int(saved_args.get("vit_layers",2)),
            num_heads=int(saved_args.get("vit_heads",4)),
            mlp_ratio=float(saved_args.get("vit_mlp_ratio",4.0)),
            p_drop=float(saved_args.get("vit_drop",0.05)),
            max_tokens=int(saved_args.get("vit_max_tokens",1024)),
            min_keep_tokens=int(saved_args.get("vit_min_keep",256)),
        ).to(device)

    # 探测 in_dim（两种路径：向量 or 4D）
    dummy_img = np.zeros((256,256,3), np.uint8)
    dummy_pts = np.array([[128,128,1.0]], np.float32)
    with torch.no_grad():
        # 尝试 4D
        img4, m4, p4 = extractor([dummy_img],[dummy_pts],[{}], return_4d=True)
        in_dim_4d = int(img4.shape[1])
        vec = (img4*m4).flatten(2).mean(-1)  # 类似池化
        in_dim = int(vec.shape[-1])
        # 同时保留 4D head 的 in_dim
        in_dim_vit = in_dim_4d

    # ===== 路径 A：E2E ViT（best_e2e.pt）=====
    if has_sam2 and has_head and (head_hint=="vit" or "vit" in str(args.ckpt) or "vit" in str(head_hint or "").lower()):
        head = _mk_head_vit(in_dim_vit)
        # SAM2 权重
        st = ckpt.get("sam2_state", ckpt)
        try:
            extractor.model.load_state_dict(st, strict=False)
        except Exception:
            pass
        # head 权重
        head.load_state_dict(ckpt["head_state"], strict=False)
        return extractor, head, "vit", id2name

    # ===== 路径 B：Finetune + 非 ViT（best_full_finetune.pt）=====
    if has_sam2 and has_head:
        # 按保存时 head 类型还原
        if head_hint == "cosine":
            head=_mk_head_cos(in_dim)
        elif head_hint == "mlp":
            head=_mk_head_mlp(in_dim, hidden=ckpt.get("args",{}).get("hidden",0))
        else:
            head=_mk_head_linear(in_dim)
        # load
        st = ckpt.get("sam2_state", ckpt)
        try:
            extractor.load_state_dict(st, strict=False)
        except Exception:
            extractor.model.load_state_dict(st, strict=False)
        head.load_state_dict(ckpt["head_state"], strict=False)
        return extractor, head, (head_hint or "finetune"), id2name

    # ===== 路径 C：冻结骨干（仅 head_state）=====
    # 需要用户/ckpt 提示 head 类型；尽量推断
    force_head = (args.force_head or (head_hint if isinstance(head_hint,str) else None))
    force_head = (force_head or "").lower()

    if force_head == "cosine" or ("weight" in ckpt and isinstance(ckpt["weight"], torch.Tensor)):
        head = _mk_head_cos(in_dim)
        state = ckpt.get("head_state", ckpt)
        head.load_state_dict(state, strict=False)
        return extractor, head, "cosine", id2name

    if force_head == "mlp":
        head = _mk_head_mlp(in_dim, hidden=ckpt.get("args",{}).get("hidden",0))
        state = ckpt.get("head_state", ckpt)
        head.load_state_dict(state, strict=False)
        return extractor, head, "mlp", id2name

    if force_head == "linear":
        head = _mk_head_linear(in_dim)
        state = ckpt.get("head_state", ckpt)
        head.load_state_dict(state, strict=False)
        return extractor, head, "linear", id2name

    # 猜测：优先 cosine -> mlp -> linear
    try:
        head = _mk_head_cos(in_dim); head.load_state_dict(ckpt.get("head_state", ckpt), strict=False)
        return extractor, head, "cosine", id2name
    except Exception:
        try:
            head = _mk_head_mlp(in_dim); head.load_state_dict(ckpt.get("head_state", ckpt), strict=False)
            return extractor, head, "mlp", id2name
        except Exception:
            head = _mk_head_linear(in_dim); head.load_state_dict(ckpt.get("head_state", ckpt), strict=False)
            return extractor, head, "linear", id2name

# =============== evaluate ===============
@torch.inference_mode()
def evaluate(extractor: nn.Module, head: nn.Module, loader: DataLoader, device: str,
             head_type: str, id2name: Dict[int,str], out_dir: Path,
             save_logits: bool = True):
    extractor.eval(); head.eval()
    y_true=[]; y_pred=[]; all_logits=[]
    pbar=tqdm(loader, total=len(loader), ncols=100, desc="Eval")
    for batch in pbar:
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device)
        if head_type=="vit":
            img4, m4, p4 = extractor(imgs, pts, metas, return_4d=True)
            logits = head(img4, m4, p4)
        else:
            vecs = extractor(imgs, pts, metas, return_4d=False)
            logits = head(vecs)
        pred = logits.argmax(1)
        y_true.extend(y.tolist()); y_pred.extend(pred.tolist())
        if save_logits:
            all_logits.append(logits.detach().cpu())
    y_true = torch.tensor(y_true); y_pred=torch.tensor(y_pred)
    metrics = _compute_metrics(y_true, y_pred, n_classes=len(id2name))
    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    # confusion png + per-class csv
    cm = metrics["confusion"].numpy()
    _save_confmat(cm, id2name, out_dir/"confusion.png",
                  title=f"acc={metrics['acc']:.3f} macroF1={metrics['macro_f1']:.3f} balAcc={metrics['balanced_acc']:.3f}")
    rows=[]
    per_cls_total=cm.sum(1); per_cls_correct=np.diag(cm)
    for c in range(cm.shape[0]):
        n=int(per_cls_total[c]); acc_c=(per_cls_correct[c]/n) if n>0 else float("nan")
        rows.append({
            "class_id": c,
            "class_name": id2name.get(c,str(c)),
            "support": n,
            "acc": float(acc_c)
        })
    pd.DataFrame(rows).to_csv(out_dir/"per_class.csv", index=False)
    with open(out_dir/"metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "acc": metrics["acc"],
            "macro_f1": metrics["macro_f1"],
            "balanced_acc": metrics["balanced_acc"],
            "n": int(len(y_true)),
            "head_type": head_type
        })+"\n")
    if save_logits and len(all_logits)>0:
        np.savez_compressed(out_dir/"logits.npz", logits=torch.cat(all_logits,0).numpy(), y=y_true.numpy(), y_pred=y_pred.numpy())
    # console
    print(f"[RESULT] acc={metrics['acc']:.4f}  macroF1={metrics['macro_f1']:.4f}  balAcc={metrics['balanced_acc']:.4f}")
    # also a text log
    with open(out_dir/"eval_log.txt","a",encoding="utf-8") as f:
        f.write(f"acc={metrics['acc']:.6f} macroF1={metrics['macro_f1']:.6f} balAcc={metrics['balanced_acc']:.6f}\n")

# =============== main ===============
def main():
    ap = argparse.ArgumentParser("Unified SAM2 classifier test (5 models, 3 datasets)")
    # 固定为相对路径
    ap.add_argument("--sam2-cfg",  type=str, default="sam2_hiera_l.yaml")
    ap.add_argument("--sam2-ckpt", type=str, default="pretrain_params/sam2_hiera_large.pt")
    # 只用一个参数切数据集
    ap.add_argument("--dataset", choices=["combined","cholect","ood5"], default="combined")
    # 可选：手动覆盖
    ap.add_argument("--manifest",  type=str, default=None)
    ap.add_argument("--label-map", type=str, default=None)
    # ckpt & 输出
    ap.add_argument("--ckpt", type=str, required=True, help="head-only / best_e2e.pt / best_full_finetune.pt")
    ap.add_argument("--out-root", type=str, default="eval_runs")
    # 其他
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-input-edge", type=int, default=None, help="限制长边，可留空")
    ap.add_argument("--force-head", choices=["linear","mlp","cosine","vit"], default=None, help="万一自动识别失败可强制指定")
    ap.add_argument("--save-logits", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # 数据集映射
    if args.manifest is None or args.label_map is None:
        if args.dataset == "combined":
            manifest_path = SMALLFILE_ROOT / "val_manifest_10.csv"
            label_map_path = SMALLFILE_ROOT / "label_map.json"
        elif args.dataset == "cholect":
            manifest_path = SMALLFILE_ROOT / "cholect" / "val_manifest_10.csv"
            label_map_path = SMALLFILE_ROOT / "cholect" / "label_map.json"
        else:  # ood5
            # 这里默认你把 5 类 OOD 的脚本输出到了这个目录
            base = Path("/home/wcheng31/sam2_classify/ood5")
            manifest_path = base / "manifest.csv"
            label_map_path = base / "label_map.json"
    else:
        manifest_path = Path(args.manifest); label_map_path = Path(args.label_map)

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map 不存在: {label_map_path}")

    # 读取 label_map / dataloader
    with open(label_map_path,"r",encoding="utf-8") as f:
        lm = json.load(f)
    tool2id = lm["tool_to_id"]
    id2name = {int(v):k for k,v in tool2id.items()}

    ds = FramePointDataset(manifest_path, label_map_path, resize=None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True, collate_fn=collate_varlen)

    # 构建 & 加载
    extractor, head, head_type, id2name_from_ckpt = _build_and_load(args, tool2id, device)
    # 以当前数据集的 id2name 为准（避免跨集 label-map 不同）
    head_type = (args.force_head or head_type)

    # 输出目录（无时间戳）
    run_name = Path(args.ckpt).stem
    out_dir = Path(args.out_root) / args.dataset / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 评估
    evaluate(extractor, head, dl, device, head_type, id2name, out_dir, save_logits=args.save_logits)

if __name__ == "__main__":
    main()

# 1) 冻结骨干 + 线性头（Frozen + Linear）
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --ckpt ckpts/frozen_linear_head.pt \
#   --dataset combined

# 2) 冻结骨干 + MLP 头（Frozen + MLP）
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --ckpt ckpts/frozen_mlp_head.pt \
#   --dataset combined

# 3) 冻结骨干 + Cosine 头（Frozen + Cosine）
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --ckpt ckpts/frozen_cosine_head.pt \
#   --dataset combined


# 4) 端到端 + ViTTokenHead（Finetune + ViT）
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --ckpt sam2_classifier/vit_head_e2e/best_e2e.pt \
#   --dataset cholect

# 5) 全量微调 + 蒸馏（Finetune + Distill，非 ViT 头）
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --ckpt sam2_classifier/distill_maskcls_t/best_full_finetune.pt \
#   --dataset ood5