#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-Anything V2 inference (filtered by prompts.json).

- 只处理 prompts.json 中列出的 frame_file
- 保存两类结果到 --out_dir:
    1) {basename}_depth16.png   (16-bit 单通道)
    2) {basename}_depth_viz.jpg (彩色可视化)

2025-08-13  by WZC
"""

import os, sys, json, argparse
from typing import List, Set
import numpy as np
import cv2
import torch
from PIL import Image

# -------------------- 兼容导入：v2 优先，v1 兜底 --------------------
def _import_da():
    """
    优先 Depth-Anything V2；若失败回退到 V1。
    返回 (infer_fn, expected_input_size)
    - infer_fn: callable(PIL.Image|np.ndarray)-> np.ndarray depth(H,W), float32
    - expected_input_size: int，模型侧的默认方形输入（常见 518）；None 表示自适应
    """
    # ---- 尝试 v2 ----
    try:
        # 大多数 v2 安装的写法（根据官方示例适配）
        from depth_anything_v2.dpt import DepthAnythingV2
        import torch.nn.functional as F

        class _V2Runner:
            def __init__(self, device, encoder: str = "vitl"):
                self.device = device
                self.model = DepthAnythingV2(encoder=encoder).to(device).eval()

            @torch.inference_mode()
            def __call__(self, img):
                # 接受 PIL 或 ndarray，转成 RGB ndarray
                if isinstance(img, Image.Image):
                    rgb = np.array(img.convert("RGB"))
                else:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 统一缩放成 518x518（Depth-Anything v2 常用）
                inp = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
                inp = torch.from_numpy(inp).float() / 255.0        # HWC, 0~1
                inp = inp.permute(2, 0, 1).unsqueeze(0).to(self.device)  # 1x3xH'xW'

                pred = self.model(inp)   # 期望输出 1x1xH'xW' 或 1xH'xW'
                if pred.ndim == 4:
                    pred = pred[:, 0]    # 1xH'xW'
                depth = pred[0].detach().cpu().numpy()  # H'xW'

                return depth

        def _infer(img, runner=[None]):
            if runner[0] is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                runner[0] = _V2Runner(device=device, encoder="vitl")
            return runner[0](img)

        return _infer, 518

    except Exception as e_v2:
        # ---- 回退 v1（如果你环境里只有 v1 包名）----
        try:
            from depth_anything.dpt import DepthAnything  # v1 包
            class _V1Runner:
                def __init__(self, device, encoder="vitl"):
                    self.device = device
                    self.model = DepthAnything(encoder=encoder).to(device).eval()

                @torch.inference_mode()
                def __call__(self, img):
                    if isinstance(img, Image.Image):
                        rgb = np.array(img.convert("RGB"))
                    else:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    inp = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
                    inp = torch.from_numpy(inp).float() / 255.0
                    inp = inp.permute(2, 0, 1).unsqueeze(0).to(self.device)

                    pred = self.model(inp)
                    if pred.ndim == 4:
                        pred = pred[:, 0]
                    depth = pred[0].detach().cpu().numpy()
                    return depth

            def _infer(img, runner=[None]):
                if runner[0] is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    runner[0] = _V1Runner(device=device, encoder="vitl")
                return runner[0](img)

            return _infer, 518

        except Exception as e_v1:
            msg = (
                "❌ 无法导入 Depth-Anything：\n"
                f"  v2 导入错误：{repr(e_v2)}\n"
                f"  v1 导入错误：{repr(e_v1)}\n"
                "请确认已正确安装官方仓库（建议 v2）：\n"
                "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
                "  cd Depth-Anything-V2 && pip install -e .\n"
            )
            print(msg, file=sys.stderr)
            sys.exit(2)

# -------------------- 只处理 prompts.json 中的帧 --------------------
def load_frame_whitelist(prompts_path: str) -> Set[str]:
    """
    支持两种格式：
    1) JSON 数组/对象，其中每个元素（或 frames 列表项）含有 frame_file
    2) JSONL（逐行 JSON），每行含有 frame_file
    """
    wanted = set()
    with open(prompts_path, "r") as f:
        txt = f.read().strip()
    # 尝试 JSON
    try:
        data = json.loads(txt)
        candidates = data if isinstance(data, list) else data.get("frames", data)
        if isinstance(candidates, dict):
            # 兼容 { "frame_id":..., "frame_file":... } 的单条情况
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

def save_depth_products(base: str, depth: np.ndarray, out_dir: str, orig_hw):
    """
    depth: H'xW' float32，范围未定。先按 min-max 归一化。
    保存：
      - {base}_depth16.png   (16-bit 单通道)
      - {base}_depth_viz.jpg (彩色图，COLORMAP_TURBO)
    """
    H0, W0 = orig_hw
    d = depth.astype(np.float32)
    d -= d.min()
    d /= (d.max() + 1e-8)

    # 恢复到原图大小（可视化/保存一致）
    d_resized = cv2.resize(d, (W0, H0), interpolation=cv2.INTER_CUBIC)

    # 16bit
    d16 = (d_resized * 65535.0).astype(np.uint16)
    p16 = os.path.join(out_dir, f"{base}_depth16.png")
    cv2.imwrite(p16, d16)

    # 伪彩
    viz_u8 = (d_resized * 255.0).astype(np.uint8)
    viz = cv2.applyColorMap(viz_u8, cv2.COLORMAP_TURBO)
    pjpg = os.path.join(out_dir, f"{base}_depth_viz.jpg")
    cv2.imwrite(pjpg, viz, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return p16, pjpg

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser("Depth-Anything v2 inference filtered by prompts.json")
    ap.add_argument("--img_dir",   type=str, required=True,
                    help="原始 RGB 图目录（例如 .../video21/ws_0/images）")
    ap.add_argument("--out_dir",   type=str, required=True,
                    help="输出目录（将写入 *_depth16.png, *_depth_viz.jpg）")
    ap.add_argument("--prompts_json", type=str, default=None,
                    help="只处理该 JSON/JSONL 中出现的 frame_file")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 导入模型
    infer_fn, exp_size = _import_da()
    print(f"[Depth-Anything] ready (expected input: {exp_size if exp_size else 'flexible'})")

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
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        img  = cv2.imread(p)            # BGR
        if img is None:
            print(f"skip(read-fail): {p}")
            continue

        depth = infer_fn(img)           # H'xW' (float32), 未归一化
        save_depth_products(base, depth, args.out_dir, orig_hw=img.shape[:2])
        print(f"{base} ✓")

    print("All done!")

if __name__ == "__main__":
    main()
