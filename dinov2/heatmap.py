#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-attention visualization for **DINO (ViT only)**
2025-08-13  by WZC

新增：
- --prompts_json：仅处理 JSON 中给出的 frame_file 列表
- 仅使用 ViTModel + safetensors
- 自动对齐 model.config.image_size，避免尺寸不匹配
"""

import os, argparse, re, json
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import cv2
from transformers import AutoImageProcessor, ViTModel

# ★ 强制使用能返注意力权重的实现（避免 sdpa/flash/xformers 路径）
os.environ.setdefault("HF_ATTENTION_BACKEND", "EAGER")
os.environ.setdefault("PYTORCH_SDP_KERNEL", "math")
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

def load_processor(model_id: str):
    print(f"Loading processor for {model_id} …")
    try:
        return AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    except TypeError:
        return AutoImageProcessor.from_pretrained(model_id)

def load_dino_vit_safetensors(model_id: str, device):
    print(f"Loading {model_id} (ViTModel, safetensors only) …")
    # ★ 优先请求 eager 注意力实现；老版本 transformers 不支持这个参数就回退
    try:
        model = ViTModel.from_pretrained(model_id, use_safetensors=True, attn_implementation="eager")
    except TypeError:
        model = ViTModel.from_pretrained(model_id, use_safetensors=True)
    model.eval().to(device)
    model.config.output_attentions = True     # ★ 保险：配置层面打开
    model.config.return_dict = True           # ★ 保险：返回字典对象
    return model

def get_patch_size(model, fallback_from_ckpt: str):
    ps = getattr(getattr(model, "config", None), "patch_size", None)
    if ps is not None:
        return int(ps)
    m = re.search(r"\d+", fallback_from_ckpt or "")
    return int(m.group()) if m else 16

def get_input_size(model):
    img_size = getattr(getattr(model, "config", None), "image_size", None)
    return int(img_size) if img_size is not None else 224

def load_frame_whitelist(prompts_path: str):
    """从 prompts.json 读取要处理的 frame_file 列表（支持 list 或 jsonl）。"""
    wanted = set()
    with open(prompts_path, "r") as f:
        txt = f.read().strip()
    try:
        data = json.loads(txt)
        candidates = data if isinstance(data, list) else data.get("frames", [])
        for item in candidates:
            fn = item.get("frame_file")
            if fn: wanted.add(os.path.basename(fn))
    except json.JSONDecodeError:
        # 逐行 JSON（容忍尾随逗号）
        for line in txt.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                item = json.loads(line)
                fn = item.get("frame_file")
                if fn: wanted.add(os.path.basename(fn))
            except Exception:
                pass
    return wanted

# ---------- 1. CLI ----------
parser = argparse.ArgumentParser("Visualize DINO (ViT) attention maps")
parser.add_argument("--image_path",
                    default="/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data/video21/ws_0/images",
                    help="RGB image or directory")
parser.add_argument("--output_dir",
                    default="/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/figures_attn_dino1",
                    help="folder to save results")
parser.add_argument("--ckpt",
                    default="/projects/surgical-video-digital-twin/pretrain_params/dino-vitb16",
                    help="HF model id or local directory, e.g. /.../pretrain_params/dino-vitb16 or facebook/dino-vitb16")
parser.add_argument("--block_id", type=int, default=-1,
                    help="which transformer block (-1=last)")
parser.add_argument("--alpha", type=float, default=0.6,
                    help="overlay weight (0-1)")
parser.add_argument("--prompts_json",
                    default=None,
                    help="只处理该 JSON 列出的 frame_file（绝对/相对文件名均可）")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Load model ----------
try:
    processor = load_processor(args.ckpt)
    model     = load_dino_vit_safetensors(args.ckpt, device)
except Exception as e:
    # 兜底到 vitb16（依然是 ViT + safetensors），防止环境变更导致跑不起来
    print(f"⚠ Failed to load '{args.ckpt}' with safetensors: {e}")
    fallback_id = "facebook/dino-vitb16"
    print(f"→ Falling back to '{fallback_id}'.")
    processor = load_processor(fallback_id)
    model     = load_dino_vit_safetensors(fallback_id, device)
    args.ckpt = fallback_id

patch_size = get_patch_size(model, args.ckpt)
input_size = get_input_size(model)
print(f"Patch size = {patch_size} | Input size = {input_size}")

# ---------- 3. Build file list (restricted by prompts.json) ----------
wanted = None
if args.prompts_json and os.path.exists(args.prompts_json):
    wanted = load_frame_whitelist(args.prompts_json)
    print(f"[prompts] loaded {len(wanted)} frame names from {args.prompts_json}")

if os.path.isdir(args.image_path):
    all_imgs = sorted([os.path.join(args.image_path, f)
                       for f in os.listdir(args.image_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
    if wanted is not None:
        img_paths = [p for p in all_imgs if os.path.basename(p) in wanted]
    else:
        img_paths = all_imgs
else:
    # 单文件：必须在 wanted 里（若提供了）
    ok = (wanted is None) or (os.path.basename(args.image_path) in wanted)
    img_paths = [args.image_path] if ok else []

os.makedirs(args.output_dir, exist_ok=True)
print(f"[run] {len(img_paths)} images to process")

# ---------- 4. main ----------
for p_img in img_paths:
    base = os.path.splitext(os.path.basename(p_img))[0]
    img_pil = Image.open(p_img).convert("RGB")
    img_pil_resized = img_pil.resize((input_size, input_size), Image.BILINEAR)
    img_np_vis = np.array(img_pil).astype(np.float32) / 255.0

    inputs = processor(img_pil_resized, return_tensors="pt", do_resize=False)
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values,
                        output_attentions=True,
                        return_dict=True,                   #
                        interpolate_pos_encoding=True)      # 旧参数，许多实现会忽略也无妨

    # ★ 若仍然拿不到 attentions，则自动兜底到 DINOv2（HF 官方实现能返注意力）
    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        from transformers import AutoModel
        print(f"⚠ Model '{args.ckpt}' did not return attentions; switching to 'facebook/dinov2-base'.")
        args.ckpt = "facebook/dinov2-base"
        processor = load_processor(args.ckpt)
        model = AutoModel.from_pretrained(args.ckpt, attn_implementation="eager").eval().to(device)
        model.config.output_attentions = True
        model.config.return_dict = True
        inputs = processor(img_pil_resized, return_tensors="pt", do_resize=False)
        pixel_values = inputs["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values, output_attentions=True, return_dict=True)

    block_id = args.block_id if args.block_id >= 0 else len(outputs.attentions) - 1
    attn = outputs.attentions[block_id]     # (1, heads, N+1, N+1)
    attn = attn[0, :, 0, 1:]                # CLS→patch

    h_feat = pixel_values.shape[-2] // patch_size
    w_feat = pixel_values.shape[-1] // patch_size
    attn = attn.reshape(attn.shape[0], h_feat, w_feat)  # (heads, H', W')

    attn_mean = attn.mean(0, keepdim=True)
    attn_up   = F.interpolate(attn_mean.unsqueeze(0), scale_factor=patch_size,
                              mode="nearest")[0, 0].cpu().numpy()

    attn_up = cv2.resize(attn_up, (img_np_vis.shape[1], img_np_vis.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

    attn_norm = (attn_up - attn_up.min()) / (1e-8 + attn_up.max() - attn_up.min())
    heatmap   = cv2.applyColorMap(np.uint8(255 * attn_norm), cv2.COLORMAP_JET)
    heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    vis = img_np_vis * (1 - args.alpha) + heatmap * args.alpha
    vis = (vis / max(vis.max(), 1e-6) * 255).astype(np.uint8)

    out_path = os.path.join(args.output_dir, f"{base}_dino_attn.jpg")
    Image.fromarray(vis).save(out_path, quality=95)
    print(f"{base} → {out_path}")

print("All done!")
