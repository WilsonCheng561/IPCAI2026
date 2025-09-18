#!/usr/bin/env python3
# coco_to_pkl_k.py  —— 传 in.json  out.pkl  K

import json, random, pickle, sys
from collections import OrderedDict
from pathlib import Path
random.seed(42)

def convert(json_in: Path, pkl_out: Path, k: int):
    data = json.loads(json_in.read_text())
    out = OrderedDict()

    # 收集 annotation 信息
    for ann in data["annotations"]:
        cid = ann["category_id"]
        out.setdefault(cid, []).append({
            "img_id": ann["image_id"],
            "ann_id": ann["id"],
            "bbox": ann.get("bbox"),
            "segmentation": ann.get("segmentation")
        })

    # 确保所有类别都存在
    all_cids = [c["id"] for c in data["categories"]]
    for cid in all_cids:
        out.setdefault(cid, [])

    # 精准取 / 补到 k
    for cid, lst in out.items():
        if len(lst) > k:
            out[cid] = random.sample(lst, k)
        elif len(lst) < k and len(lst) > 0:
            out[cid].extend(random.choices(lst, k=k-len(lst)))
        elif len(lst) == 0:
            out[cid] = [{"img_id": None, "ann_id": None, "bbox": None, "segmentation": None}] * k

    with pkl_out.open("wb") as f:
        pickle.dump(out, f)
    print(f"✅ {pkl_out} 生成完毕，每类 {k} 条")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python coco_to_pkl_k.py in.json out.pkl K")
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
