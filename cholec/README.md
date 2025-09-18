1. **build_cholec80_custom_dataset.py**  
   - 通过改变 video index 从mnt来增量制作自己的 cholec80 数据集，另外注意T/F和F/T来制作reference和targets，将之前的 prompt.json 转换为 COCO 格式注释custom_references.json，以及 bbox 的可视化  

   限制于CPU速度只能一个一个test的话
   python /home/wcheng31/no-time-to-train/cholec/slice_coco_by_video.py \
   --in /projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/data/annotations/custom_targets_with_SAM_segm.json \
  --videos 21

2. **sam_bbox_to_segm_batch.py**  
   - INPUT_JSON 注意跑两次
   - 加载 COCO 格式的 bbox 标注来对每一张参考图像进行 mask 推理，每个掩码转成 COCO 格式的 segmentation（即 polygon，新的文件叫做 `custom_references_with_SAM_segm.json`，是 1 中 `custom_references.json` 的升级版本，在原始的 bbox 上加上了 segmentation 信息，用于训练分割模型或者做图像检索  
   - 额外可视化了一下分割后的图像，保存在形如  
     `/projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/data/annotations/references_visualisations/video01_0000000_segm.png`  
     python /home/wcheng31/no-time-to-train/cholec/sam_bbox_to_segm_batch.py


3. **coco_to_pkl_k.py**  
   - Convert coco annotations to pickle file，**需要 1-shot 或 10-shot 时，只改 `TARGET_K` 即可**  
   python /home/haoding/Wenzheng/no-time-to-train/cholec/coco_to_pkl_k.py

4. **run_lightening.py** 
   - fill_memory：把 K 张/条参考图像丢进模型，抽出它们的特征，写进记忆库，输出 *_refs_memory.pth
   - postprocess_memory：对刚写好的特征做聚类 / 去重，让记忆库更精简，输出 *_refs_memory_post.pth
   - test： 带着记忆库跑在 target 图片上，输出分割 + mAP 分数， 分数会打印在终端，同时 CSVLogger 落到 work_dirs/.../version_x/metrics.csv

   - .pth 里有什么？冻结的 SAM-2、DINOv2 权重、memory_bank（10 类 × K 条 prototype 特征 + 相关掩码）

   - 在终端一次性跑多个K值
   - **bash /home/haoding/Wenzheng/no-time-to-train/cholec/run.sh 1 5 10 20 50 100**
   - **bash /home/haoding/Wenzheng/no-time-to-train/cholec/run.sh 5**
   - **bash /home/haoding/Wenzheng/no-time-to-train/cholec/run_one_video.sh 21 5**
   - (第一个参数: 视频如 21 或 21-23;第二个参数: K-shot, 默认 5)
