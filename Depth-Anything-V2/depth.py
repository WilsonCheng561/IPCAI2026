#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-Anything V2 inference (filtered by prompts.json) - Fixed version
2025-08-19: Fixed to match official implementation
"""

import os, sys, json, argparse
from typing import List, Set
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib
import glob

# -------------------- 使用官方实现 --------------------
def load_depth_anything_model(encoder='vitl', device='cuda'):
    """加载Depth-Anything模型（官方实现）"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        print(f"[Model] Loading DepthAnythingV2 with {encoder} encoder")
        
        # 加载模型
        model = DepthAnythingV2(**model_configs[encoder])
        
        # 加载预训练权重 - 使用绝对路径确保找到权重文件
        ckpt_path = f"/projects/surgical-video-digital-twin/pretrain_params/depth_anything_v2_{encoder}.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
        
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        sys.exit(1)

def infer_depth(model, image, input_size=518, device='cuda'):
    """使用官方方法推理深度图"""
    # 官方预处理
    h, w = image.shape[:2]
    
    # 调整大小保持宽高比
    scale = input_size / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 确保能被14整除（ViT要求）
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14
    
    # 调整大小
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 转换为Tensor
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    image = image.unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        depth = model(image)
        if depth.ndim == 4:
            depth = depth[:, 0]
        depth = depth[0].detach().cpu().numpy()
    
    # 恢复到原始尺寸
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    return depth

# -------------------- 只处理 prompts.json 中的帧 --------------------
def load_frame_whitelist(prompts_path: str) -> Set[str]:
    wanted = set()
    with open(prompts_path, "r") as f:
        txt = f.read().strip()
    
    # 尝试 JSON
    try:
        data = json.loads(txt)
        candidates = data if isinstance(data, list) else data.get("frames", data)
        if isinstance(candidates, dict):
            candidates = [candidates]
        for item in candidates or []:
            fn = item.get("frame_file")
            if fn:
                wanted.add(os.path.basename(fn))
        if wanted:
            return wanted
    except json.JSONDecodeError:
        pass
    
    # 尝试 JSONL
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            item = json.loads(line)
            fn = item.get("frame_file")
            if fn:
                wanted.add(os.path.basename(fn))
        except Exception:
            continue
    
    return wanted

# -------------------- I/O & 保存 --------------------
def list_images_filtered(img_dir: str, wanted: Set[str]) -> List[str]:
    all_imgs = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )
    if wanted:
        all_imgs = [f for f in all_imgs if f in wanted]
    return [os.path.join(img_dir, f) for f in all_imgs]

def save_depth_products(base: str, depth: np.ndarray, out_dir: str):
    """
    保存深度图结果
    - {base}_depth16.png (16-bit 单通道)
    - {base}_depth_viz.jpg (彩色图)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存16-bit PNG
    depth_min = depth.min()
    depth_max = depth.max()
    
    if depth_max - depth_min < 1e-6:
        print(f"  WARNING: {base} has invalid depth (all zeros). Skipping.")
        return None, None
    
    # 归一化到0-65535范围
    depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 65535.0
    depth_16bit = depth_normalized.astype(np.uint16)
    
    p16 = os.path.join(out_dir, f"{base}_depth16.png")
    cv2.imwrite(p16, depth_16bit)
    
    # 创建伪彩色可视化
    depth_normalized_8bit = (depth_normalized / 256).astype(np.uint8)
    viz = cv2.applyColorMap(depth_normalized_8bit, cv2.COLORMAP_TURBO)
    pjpg = os.path.join(out_dir, f"{base}_depth_viz.jpg")
    cv2.imwrite(pjpg, viz)
    
    return p16, pjpg

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser("Depth-Anything v2 inference (fixed)")
    ap.add_argument("--img_dir", type=str, required=True, help="原始 RGB 图目录")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录")
    ap.add_argument("--prompts_json", type=str, default=None, help="只处理该 JSON/JSONL 中出现的 frame_file")
    ap.add_argument("--encoder", type=str, default="vitl", choices=['vits', 'vitb', 'vitl', 'vitg'], 
                   help="模型编码器类型")
    ap.add_argument("--input_size", type=int, default=518, help="模型输入尺寸")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using {device}")
    
    # 加载模型
    model = load_depth_anything_model(encoder=args.encoder, device=device)
    
    # 读取要处理的帧
    wanted = set()
    if args.prompts_json and os.path.exists(args.prompts_json):
        wanted = load_frame_whitelist(args.prompts_json)
        print(f"[prompts] loaded {len(wanted)} frames from {args.prompts_json}")
    else:
        print("[prompts] not provided → will process ALL images in img_dir")
    
    img_paths = list_images_filtered(args.img_dir, wanted)
    print(f"[run] {len(img_paths)} images to process")
    
    # 主循环
    processed_count = 0
    skipped_count = 0
    
    for i, p in enumerate(img_paths):
        base = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)  # BGR格式
        
        if img is None:
            print(f"[{i+1}/{len(img_paths)}] skip(read-fail): {p}")
            skipped_count += 1
            continue
        
        # 检查图像尺寸和通道
        print(f"[{i+1}/{len(img_paths)}] {base} | shape: {img.shape} | channels: {img.shape[2] if len(img.shape)>2 else 1}")
        
        try:
            # 推理深度图（使用官方方法）
            depth = infer_depth(model, img, input_size=args.input_size, device=device)
            
            # 打印深度图统计信息
            print(f"  depth raw: min={depth.min():.6f}, max={depth.max():.6f}, mean={depth.mean():.6f}")
            
            # 保存结果
            depth_path, viz_path = save_depth_products(base, depth, args.out_dir)
            
            if depth_path and viz_path:
                processed_count += 1
                print(f"  ✓ saved depth: {os.path.basename(depth_path)} | viz: {os.path.basename(viz_path)}")
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ERROR during inference: {str(e)}")
            skipped_count += 1
    
    print("\nProcessing summary:")
    print(f"  Total images: {len(img_paths)}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print("All done!")

if __name__ == "__main__":
    main()
