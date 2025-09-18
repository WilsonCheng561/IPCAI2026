#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把一个 COCO 标注文件按视频号切分，只保留 file_name 以 videoXX_ 开头的图片与对应的 anns。
用法:
  python /home/haoding/Wenzheng/no-time-to-train/cholec/slice_coco_by_video.py \
      --in /mnt/disk0/haoding/no-time-to-train/data/annotations/custom_targets_with_SAM_segm.json \
      --videos 21            # 只保留 video21_*
  python tools/slice_coco_by_video.py --in ... --videos 21-23
  python tools/slice_coco_by_video.py --in ... --videos 21,22,30
"""
import json, re
from pathlib import Path
import argparse

# <<< ADDED: 统一默认输出目录到你的 no-time-to-train 工作区
DEFAULT_OUT_DIR = Path(
    "/projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/data/annotations"
)

def parse_videos(spec: str):
    spec = spec.strip().lower().replace("video", "")
    vids = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            vids |= {f"{i:02d}" for i in range(a, b+1)}
        else:
            vids.add(f"{int(part):02d}")
    return sorted(vids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_json", required=True)
    ap.add_argument("--videos", required=True, help="如 21 或 21-23 或 21,22,30")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_json)
    data = json.loads(in_path.read_text())

    vids = parse_videos(args.videos)
    prefixes = tuple([f"video{v}_" for v in vids])

    keep_images = [im for im in data["images"] if im["file_name"].startswith(prefixes)]
    keep_ids = {im["id"] for im in keep_images}

    keep_anns = [ann for ann in data.get("annotations", []) if ann["image_id"] in keep_ids]
    # 类别直接透传（或者只保留出现过的也行）
    cats = data.get("categories", [])

    out = {
        "images": keep_images,
        "annotations": keep_anns,
        "categories": cats,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # <<< ADDED: 确保目录存在
    else:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)  # <<< ADDED
        out_name = f"{in_path.stem}_v{','.join(vids)}.json"
        out_path = DEFAULT_OUT_DIR / out_name

    out_path.write_text(json.dumps(out))
    print(f"✅ Wrote {out_path} (images={len(keep_images)}, anns={len(keep_anns)})")

if __name__ == "__main__":
    main()
