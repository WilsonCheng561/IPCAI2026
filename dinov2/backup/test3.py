#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch attention‑map visualizer  ——  DINO‑v2 (HF only, no torch.hub)
2025‑07‑25  •  Haoding custom path version
"""

import os, warnings, cv2, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from transformers import Dinov2Model, AutoImageProcessor

# ────────── 0. 关闭无关警告 ──────────
warnings.filterwarnings("ignore", message="The torchvision.*v2")
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

# ────────── 1. 硬编码参数 ──────────
IMAGE_DIR  = "/home/haoding/Wenzheng/dinov2/figures"
OUTPUT_DIR = "/home/haoding/Wenzheng/dinov2/figures_attn_dinov2"
ARCH       = "dinov2_vitg14"          #  vitb14 / vitl14 / vitg14
BLOCK_ID   = -1                       #  -1 = last transformer layer
MODE       = "patch2cls"              #  cls2patch  |  patch2cls
ALPHA      = 0.6                      #  overlay weight

device = "cuda" if torch.cuda.is_available() else "cpu"

# ────────── 2. HF 权重映射 ──────────
hf_map = {
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
}
if ARCH not in hf_map:
    raise ValueError(f"ARCH 必须选 {list(hf_map.keys())}")

print(f"Loading {hf_map[ARCH]}  …")
processor = AutoImageProcessor.from_pretrained(hf_map[ARCH])
model     = Dinov2Model.from_pretrained(hf_map[ARCH]).eval().to(device)
patch_size = model.config.patch_size        # =14

# ────────── 3. 收集图片 ──────────
img_paths = sorted([os.path.join(IMAGE_DIR, f)
                    for f in os.listdir(IMAGE_DIR)
                    if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
os.makedirs(OUTPUT_DIR, exist_ok=True)

for p in img_paths:
    name = os.path.splitext(os.path.basename(p))[0]
    pil_img = Image.open(p).convert("RGB")
    np_img  = np.asarray(pil_img).astype(np.float32) / 255.0

    # ---- 预处理（保持 518×518，可整除 14）----
    inputs = processor(pil_img, size=518, crop_size=518, return_tensors="pt")
    pixel  = inputs["pixel_values"].to(device)    # (1,3,518,518)

    # ---- 前向 + 取 attentions ----
    with torch.no_grad():
        outs = model(pixel_values=pixel, output_attentions=True)

    layer = outs.attentions[BLOCK_ID]  # (1, heads, N+1, N+1)
    if MODE == "cls2patch":
        att = layer[0,:,0,1:]          # CLS → patch
    elif MODE == "patch2cls":
        att = layer[0,:,1:,0]          # patch → CLS
    else:
        raise ValueError("MODE 只能是 cls2patch 或 patch2cls")

    Hf = pixel.shape[-2] // patch_size   # 37
    Wf = pixel.shape[-1] // patch_size
    att = att.reshape(att.shape[0], Hf, Wf).mean(0, keepdim=True)  # (1,Hf,Wf)

    att_up = F.interpolate(att.unsqueeze(0), scale_factor=patch_size,
                           mode="nearest")[0,0].cpu().numpy()      # (518,518)

    # ---- resize 到原图尺寸 ----
    att_up = cv2.resize(att_up, (np_img.shape[1], np_img.shape[0]), cv2.INTER_CUBIC)
    norm   = (att_up - att_up.min()) / (att_up.ptp() + 1e-8)
    heat   = cv2.cvtColor(cv2.applyColorMap((norm*255).astype(np.uint8),
                                            cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)/255.0
    vis    = np.clip(np_img*(1-ALPHA) + heat*ALPHA, 0, 1)
    out    = os.path.join(OUTPUT_DIR, f"{name}_{ARCH}_{MODE}.jpg")
    Image.fromarray((vis*255).astype(np.uint8)).save(out, quality=95)
    print(f"{name}  →  {out}")

print("全部完成！")
