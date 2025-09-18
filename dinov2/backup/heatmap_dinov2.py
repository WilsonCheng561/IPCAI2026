#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch‑attention visualization for **DINOv2 (ViT)**
2025‑07‑22  by WZC
"""

import os, argparse, re, random, colorsys
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F, torchvision
from torchvision import transforms as T
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn

# ---------- 1. CLI ----------
parser = argparse.ArgumentParser("Visualize DINOv2 attention maps")
parser.add_argument("--image_path",
                    default="/home/haoding/Wenzheng/dinov2/figures",
                    help="RGB image or directory")
parser.add_argument("--output_dir",
                    default="/home/haoding/Wenzheng/dinov2/figures_attn_dino2",
                    help="folder to save results")
parser.add_argument("--model_type",
                    default="vit_large",
                    choices=["vit_small", "vit_base", "vit_large", "vit_giant"],
                    help="DINOv2 model type")
parser.add_argument("--block_id", type=int, default=-1,
                    help="which transformer block (-1=last)")
parser.add_argument("--alpha", type=float, default=0.6,
                    help="overlay weight (0‑1)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Load DINOv2 Model ----------
print(f"Loading DINOv2 {args.model_type} ...")
dinov2_model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{args.model_type}')
dinov2_model.eval().to(device)

# Get patch size based on model type
patch_size = {
    "vit_small": 14,
    "vit_base": 14,
    "vit_large": 14,
    "vit_giant": 14
}[args.model_type]

# ---------- 3. I/O ----------
if os.path.isdir(args.image_path):
    img_paths = sorted([os.path.join(args.image_path, f)
                        for f in os.listdir(args.image_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
else:
    img_paths = [args.image_path]

os.makedirs(args.output_dir, exist_ok=True)

# ---------- 4. DINOv2 Attention Extraction ----------
def forward_selfattention(image, model, block_id=-1):
    """
    Extract attention maps from DINOv2 model
    """
    with torch.no_grad():
        # Forward pass through the model
        outputs = model.get_intermediate_layers(
            image,
            n=[block_id] if block_id >= 0 else None,
            reshape=True,
            return_class_token=True,
            norm=True
        )
    
    # For a single block
    if block_id >= 0:
        output = outputs[0]
    # For the last block if block_id is -1
    else:
        output = outputs[-1]
    
    # Extract keys and queries
    qkv = output[1][1]
    
    # Reshape to separate heads
    B, N, C = qkv.shape
    num_heads = model.blocks[0].attn.num_heads
    head_dim = C // (3 * num_heads)
    
    # Separate Q, K, V
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    attn_probs = attn_scores.softmax(dim=-1)
    
    return attn_probs

# ---------- 5. Main Processing ----------
for p_img in img_paths:
    base = os.path.splitext(os.path.basename(p_img))[0]
    img_pil = Image.open(p_img).convert("RGB")
    img_np  = np.array(img_pil).astype(np.float32) / 255.0
    
    # Preprocess image for DINOv2
    transform = T.Compose([
        T.Resize(224, 224),  # DINOv2 requires 224x224 input
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Get attention maps
    block_id = args.block_id if args.block_id >= 0 else len(dinov2_model.blocks) - 1
    attn = forward_selfattention(img_tensor, dinov2_model, block_id)
    
    # Extract CLS token attention to patches
    attn = attn[0, :, 0, 1:]  # (heads, num_patches)
    
    # Reshape to patch grid
    h_feat = w_feat = int(np.sqrt(attn.shape[1]))
    attn = attn.reshape(attn.shape[0], h_feat, w_feat)  # (heads, H', W')
    
    # Average over heads
    attn_mean = attn.mean(0, keepdim=True)  # (1, H', W')
    
    # Upsample to input size (224x224)
    attn_up = F.interpolate(
        attn_mean.unsqueeze(0), 
        scale_factor=patch_size,  # patch_size = 14 for DINOv2
        mode="nearest"
    )[0, 0].cpu().numpy()  # (224,224)
    
    # Resize back to original image size
    attn_up = cv2.resize(
        attn_up, 
        (img_np.shape[1], img_np.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Normalize and apply colormap
    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
    heatmap   = cv2.applyColorMap(np.uint8(255 * attn_norm), cv2.COLORMAP_JET)
    heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    # Overlay on original image
    vis = img_np * (1 - args.alpha) + heatmap * args.alpha
    vis = (vis / vis.max() * 255).astype(np.uint8)

    # Save result
    out_path = os.path.join(args.output_dir, f"{base}_dinov2_attn.jpg")
    Image.fromarray(vis).save(out_path, quality=95)
    print(f"{base} → {out_path}")
