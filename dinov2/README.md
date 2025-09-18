# IPCAI2026: SAM2 + DINOv2 + DepthAnythingV2 Pipeline

## 🔥 Clone this repo
> [!IMPORTANT]
> Clone with SSH (make sure your GitHub key is configured):
```bash
git clone git@github.com:WilsonCheng561/IPCAI2026.git
cd IPCAI2026
``````


# 1 Environment

You can reproduce the environment:
## 1) Create the conda environment from explicit lock
``````
conda create --name sam2 --file conda-linux-64.lock

conda activate sam2

pip install -r pip-freeze.txt
``````

# 2 Data & Checkpoints(Optional)

Cholec80 root (example):
/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data/

SAM2 checkpoint & cfg:

--ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt

--cfg sam2_hiera_l.yaml

# 3 End-to-End Pipeline
Below are exact commands mirroring  workflow. Replace video IDs / paths as needed.

## Step 1 — Extract Depth & DINO features
### Depth-Anything V2
``````
python /home/wcheng31/Depth-Anything-V2/depth.py
``````
### DINOv2 heatmaps
``````
python /home/wcheng31/dinov2/heatmap.py
``````

Outputs should be written where those scripts expect (check those scripts’ args/path configs if you need to customize).

## Step 2 — Fuse features to auto-generate prompt points
``````
python /home/wcheng31/dinov2/generate_point4sam2.py
``````

This script takes the depth + dino outputs and writes point prompts to a directory like:

/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/points


(Adjust to your actual output directory in the script or via CLI arguments if available.)

## Step 3 — Run SAM2 point-prompt inference to get masks

``````
python /home/wcheng31/SAM2/run_sam2_point_infer.py \
  --pts_dir /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/points \
  --out_dir /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/figures_points_sam2 \
  --ckpt    /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
  --cfg     sam2_hiera_l.yaml
``````

This writes mask visualizations/NPYs (depending on your script settings) under --out_dir.


We already have a convenience script. Use it with any set of video IDs:

## Step 4 -You can skip step 1-3 and only run below for multiple videos:
``````
bash /home/wcheng31/dinov2/run_one_video.sh 22 23 24 25
``````

## Step 5 — Evaluate under different IoU thresholds
``````
python /home/wcheng31/dinov2/eval.py \
  --root /projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data \
  --v_start 21 --v_end 30 \
  --ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
  --cfg sam2_hiera_l.yaml \
  --iou_thr 0.8 \
  --tmp-root /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/tmp_eval \
  --out-root /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/eval
``````

--v_start/--v_end selects the video range to evaluate (inclusive/exclusive depending on your implementation; follow the script’s help).

--iou_thr sets the IoU threshold for a correct prediction.

Results and any temporary files are written to --out-root and --tmp-root.
