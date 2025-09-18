import os
import glob
import time

import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from sklearn.decomposition import PCA  # PCA 已注释掉

#tokens.pt 

#切换到dinov2的环境再用，clone本地仓库，从本地加载防止http rate limit：
#git clone https://github.com/facebookresearch/dinov2.git ~/dinov2
#一定是先 cd /home/haoding/Wenzheng/dinov2
#然后python test.py
#这里更改输入图片路径和输出路径

input_dir = "/home/haoding/Wenzheng/dinov2/figures/images"
output_dir = "/home/haoding/Wenzheng/dinov2/figures/features"  # 只保存 tokens.pt
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_dir, "*")))

# ------------------------------
# GPU 显存打印函数
# ------------------------------
def get_gpu_mem(print_prefix=""):
    """
    打印当前与峰值显存使用情况 (MB)。
    如果没有 GPU 或 torch.cuda.is_available() == False，则直接返回。
    """
    if not torch.cuda.is_available():
        return
    current_allocated = torch.cuda.memory_allocated() / 1024**2
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{print_prefix} current: {current_allocated:.2f} MB, peak: {peak_allocated:.2f} MB")

# ------------------------------
# 设备选择
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ------------------------------
# 加载模型,下面是可以选择的模型，详情见https://github.com/facebookresearch/dinov2，如果速度太慢，就选择参数量更小的
# ------------------------------
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

try:
    # model = torch.hub.load("", "dinov2_vits14", source="local")  #如果使用batch size=64，大概0.00015s一张
    # model.eval()

    # 确保提前 git clone https://github.com/facebookresearch/dinov2.git ~/dinov2
    model = torch.hub.load("/home/haoding/dinov2", "dinov2_vitg14", source="local") #这是最强的模型，如果使用batch size=64，大概0.0005s一张
    model.eval()

    if torch.cuda.is_available():
        model = model.to(torch.float16).to(device)
    else:
        pass
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        model = model.to(torch.float32)
        device = torch.device("cpu")
    else:
        raise e

# ------------------------------
# 图像预处理
# ------------------------------
patch_h = 16
patch_w = 16
feat_dim = 384

transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ------------------------------
# 定义批处理函数
# ------------------------------
def chunker(lst, chunk_size):
    """ 将列表 lst 按 chunk_size 大小分块，生成器形式返回。 """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

# ------------------------------
# 批量处理文件
# ------------------------------

batch_size = 64

for chunk_idx, chunk_paths in enumerate(chunker(image_paths, batch_size), start=1):
    # 先把本批次的图像加载并变换后堆叠起来
    batch_tensors = []
    valid_paths = []  # 存放本批次实际的图像路径（过滤非图像文件）
    for img_path in chunk_paths:
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # (1,3,H,W)
        batch_tensors.append(img_tensor)
        valid_paths.append(img_path)

    if len(valid_paths) == 0:
        continue

    # 拼成一个 batch (B,3,H,W)
    batch_input = torch.cat(batch_tensors, dim=0)

    # 转到 GPU 并半精度（如可用）
    if device.type == "cuda":
        batch_input = batch_input.to(torch.float16).to(device)

    # 清理显存统计
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --------------- 打印推理前显存 ---------------
    if torch.cuda.is_available():
        get_gpu_mem(f"[Batch {chunk_idx} Before Inference]")

    # ------------------------------
    # 推理并记录时间
    # ------------------------------
    start_time = time.time()
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                features_dict = model.forward_features(batch_input)
        else:
            features_dict = model.forward_features(batch_input)
    end_time = time.time()

    # --------------- 打印推理后显存 ---------------
    if torch.cuda.is_available():
        get_gpu_mem(f"[Batch {chunk_idx} After Inference]")

    tokens = features_dict["x_norm_patchtokens"]  # (B, N, feat_dim)
    tokens = tokens.float().cpu()                 # 转回 float32, CPU

    # 批次平均推理耗时
    batch_time = end_time - start_time
    time_per_image = batch_time / len(valid_paths)

    # ------------------------------
    # 保存 tokens + 打印时间
    # ------------------------------
    for i, path in enumerate(valid_paths):
        # 按名称生成保存路径
        filename = os.path.basename(path)
        base_name, ext = os.path.splitext(filename)
        save_name = f"{base_name}_tokens.pt"
        output_path = os.path.join(output_dir, save_name)

        # 取出该图在 batch 里的 tokens => (N, feat_dim)
        single_tokens = tokens[i]  # shape: (N, feat_dim)

        # 保存
        torch.save(single_tokens, output_path)

        # 打印处理该图像的耗时
        print(f"{path} -> {time_per_image:.8f} seconds")
