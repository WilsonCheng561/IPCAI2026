1.提取depth和dino特征
python /home/wcheng31/Depth-Anything-V2/depth.py
python /home/wcheng31/dinov2/heatmap.py

2.合并两种特征自动生成的prompt点
python /home/wcheng31/dinov2/generate_point4sam2.py

3.使用点进行infer生成seg mask
python /home/wcheng31/SAM2/run_sam2_point_infer.py       --pts_dir /home/haoding/Wenzheng/dinov2/points       --out_dir /home/haoding/Wenzheng/dinov2/figures_points_sam2

python /home/wcheng31/SAM2/run_sam2_point_infer.py \
  --pts_dir /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/points \
  --out_dir /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/figures_points_sam2 \
  --ckpt    /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
  --cfg     sam2_hiera_l.yaml


4.把所有的如上脚本统一，每次跑一个或者多个video的
bash /home/wcheng31/dinov2/run_one_video.sh 22 23 24 25
bash /home/wcheng31/dinov2/run_one_video.sh 21 22 23 24 26 27 28 29 30

5.最后是评估不同iou下我们的方法表现

python /home/wcheng31/dinov2/eval.py \
  --root /projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data \
  --v_start 21 --v_end 30 \
  --ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
  --cfg sam2_hiera_l.yaml \
  --iou_thr 0.8 \
  --tmp-root /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/tmp_eval \
  --out-root /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/eval


