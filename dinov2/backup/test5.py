import os, glob, time, math, cv2
import torch, torchvision.transforms as T, numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# ---------------- 路径 ----------------
input_dir  = "/home/haoding/Wenzheng/dinov2/figures"
output_dir = "/home/haoding/Wenzheng/dinov2/figures_pca"
os.makedirs(output_dir, exist_ok=True)
image_paths = sorted(glob.glob(os.path.join(input_dir, "*")))

# ---------------- 设备 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# ---------------- 加载模型 ----------------
model = torch.hub.load("/home/haoding/dinov2", "dinov2_vitg14", source="local")
model.eval()
if torch.cuda.is_available():
    model = model.to(torch.float16).to(device)

# ---------------- 参数 (基于模型自动推断) -----------
patch_sz  = model.patch_embed.patch_size[0]          # >>> FIX 1: 14
grid_size = 16                                       # 16×16 token grid
input_px  = patch_sz * grid_size                     # 224 px

transform = T.Compose([
    T.GaussianBlur(9,(0.1,2.0)),
    T.Resize((input_px, input_px)),                  # >>> FIX 2: 224
    T.CenterCrop((input_px, input_px)),              # 同上
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

def chunker(lst,n):
    for i in range(0,len(lst),n): yield lst[i:i+n]

# ---------- PCA 可视化 ---------- 
def tokens_to_pca_image(tokens_np, out_size):
    N = tokens_np.shape[0]
    grid = int(math.sqrt(N))
    pca  = PCA(n_components=3).fit_transform(tokens_np)
    pca_norm = (pca - pca.min(0)) / (pca.max(0)-pca.min(0)+1e-6)
    rgb = (pca_norm*255).astype(np.uint8).reshape(grid,grid,3)

    # 高质量上采样 + 模糊
    rgb_up = cv2.resize(rgb, out_size[::-1], interpolation=cv2.INTER_LANCZOS4)
    rgb_up = cv2.GaussianBlur(rgb_up, (0,0), sigmaX=1.2)
    return rgb_up

# ---------- 主循环 ----------
batch_size = 64
for cidx, paths in enumerate(chunker(image_paths,batch_size),1):
    batch_t, orig, val = [], [], []
    for p in paths:
        if not p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")): continue
        img = Image.open(p).convert("RGB")
        proc = transform.transforms[1](img)          # resize
        proc = transform.transforms[2](proc)         # crop
        orig.append(proc)
        batch_t.append(transform(img).unsqueeze(0))
        val.append(p)
    if not val: continue

    batch = torch.cat(batch_t)
    if device.type=="cuda": batch = batch.to(torch.float16).to(device)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):   # >>> FIX 3
            feats = model.forward_features(batch)
    tokens = feats["x_norm_patchtokens"].float().cpu()               # (B,N,D)

    for i,p in enumerate(val):
        stem = os.path.splitext(os.path.basename(p))[0]
        pca_rgb = tokens_to_pca_image(tokens[i].numpy(), orig[i].size)

        w,h = orig[i].size
        canvas = Image.new("RGB",(w*2,h))
        canvas.paste(orig[i],(0,0))
        canvas.paste(Image.fromarray(pca_rgb),(w,0))
        out_p = os.path.join(output_dir,f"{stem}_pca.jpg")
        canvas.save(out_p, quality=95)
        print(f"{p} -> saved {out_p}")                              # >>> FIX 4: 简洁日志