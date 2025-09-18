import os
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure, morphology

def generate_bboxes_from_attention(attn_map,
                                   min_area=100,
                                   max_boxes=5,
                                   min_wh=10):
    # 1‑2  阈值 & 形态学
    binary_mask   = adaptive_thresholding(attn_map)
    enhanced_mask = morphological_enhancement(binary_mask)

    # 3  连通域 → items
    items = connected_component_analysis(enhanced_mask,
                                         attn_map,
                                         min_area)

    # 3.5  合并邻近框
    items = merge_adjacent_boxes(items, gap=15)

    # 3.6  去掉特别小的框
    items = [it for it in items
             if (it['box'][2]-it['box'][0] >= min_wh and
                 it['box'][3]-it['box'][1] >= min_wh)]

    # 4  NMS
    items = filter_overlapping_boxes(items, iou_threshold=0.2)

    # 5  截断上限并返回
    return items[:max_boxes]


def adaptive_thresholding(attn_map):
    """自适应阈值分割"""
    # 基于注意力值分布计算动态阈值
    mean_val = np.mean(attn_map)
    std_val = np.std(attn_map)
    threshold = mean_val + 0.5 * std_val
    
    # 创建二值掩码
    binary_mask = np.zeros_like(attn_map, dtype=np.uint8)
    binary_mask[attn_map > threshold] = 1
    
    return binary_mask

# def morphological_enhancement(mask):
#     """形态学操作增强连通性"""
#     # 填充小孔洞
#     filled = morphology.remove_small_holes(mask.astype(bool), area_threshold=50)
    
#     # 闭运算连接邻近区域
#     kernel = np.ones((5, 5), np.uint8)
#     closed = morphology.closing(filled, kernel)
    
#     # 开运算去除小噪点
#     opened = morphology.opening(closed, kernel)
    
#     return opened.astype(np.uint8)

def morphological_enhancement(mask):
    """Morph‑ops: fill holes → close gaps → open noise → dilate 1 patch."""
    # ① small hole filling (tool cores often hollow)
    filled = morphology.remove_small_holes(mask.astype(bool), area_threshold=80)

    # ② closing to bridge neighbouring patches
    kernel = np.ones((7, 7), np.uint8)              # bigger than before
    closed = cv2.morphologyEx(filled.astype(np.uint8),
                              cv2.MORPH_CLOSE, kernel, iterations=2)

    # ③ opening to knock out tiny noise clusters
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # ④ extra dilation so contiguous parts fuse into one component
    dilated = cv2.dilate(opened, kernel, iterations=1)

    return dilated.astype(np.uint8)



def connected_component_analysis(mask, attn_map, min_area):
    """连通域分析 ‑> [{'box':[x0,y0,x1,y1], 'score':conf}, …]"""
    labels   = measure.label(mask)
    regions  = measure.regionprops(labels)

    results = []
    for region in regions:
        if region.area < min_area:        # 过滤小区域
            continue
        minr, minc, maxr, maxc = region.bbox
        # 置信度 = 区域平均注意力
        conf = float(attn_map[region.coords[:, 0], region.coords[:, 1]].mean())
        results.append({'box':[minc, minr, maxc, maxr], 'score':conf})

    # 按置信度排序（高→低）
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def filter_overlapping_boxes(items, iou_threshold=0.2):
    if not items: return items
    kept = [items[0]]
    for it in items[1:]:
        keep = True
        for ref in kept:
            if calculate_iou(it['box'], ref['box']) > iou_threshold:
                keep = False
                break
        if keep:
            kept.append(it)
    return kept


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 解包坐标
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # 检查是否有交集
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    # 计算交集面积
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    return inter_area / union_area if union_area > 0 else 0

def merge_adjacent_boxes(items, gap=15):
    """
    items: [{'box':[…], 'score':…}]
    返回已合并的新 items
    """
    merged = []
    for it in items:
        x0,y0,x1,y1 = it['box']; s = it['score']
        merged_flag = False
        for m in merged:
            a0,a1,b0,b1 = m['box']
            if (x0 <= b0 + gap and x1 >= a0 - gap and
                y0 <= b1 + gap and y1 >= a1 - gap):
                # union & keep max score
                m['box'] = [min(a0,x0), min(a1,y0), max(b0,x1), max(b1,y1)]
                m['score'] = max(m['score'], s)
                merged_flag = True
                break
        if not merged_flag:
            merged.append({'box':[x0,y0,x1,y1], 'score':s})
    return merged



def draw_bboxes_on_image(image, items, color=(0,255,0), thickness=3):
    img = image.copy()
    for idx,it in enumerate(items):
        x0,y0,x1,y1 = it['box']
        cv2.rectangle(img, (x0,y0), (x1,y1), color, thickness)
        cv2.putText(img, f"{idx+1}:{it['score']:.2f}",
                    (x0, y0-6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return img

import json 
def process_images(original_dir, attn_dir, output_dir, min_area=100, max_boxes=5):
    """处理所有图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取原始图像列表
    image_files = [f for f in os.listdir(original_dir) 
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        print(f"Processing: {img_file}")
        
        # 加载原始图像
        orig_path = os.path.join(original_dir, img_file)
        orig_img = cv2.imread(orig_path)
        if orig_img is None:
            print(f"Error loading original image: {orig_path}")
            continue
        
        # 查找对应的注意力图
        attn_path = os.path.join(attn_dir, f"{base_name}_dino_attn.jpg")
        if not os.path.exists(attn_path):
            print(f"Attention map not found: {attn_path}")
            continue
        
        # 加载注意力图
        attn_img = cv2.imread(attn_path)
        if attn_img is None:
            print(f"Error loading attention map: {attn_path}")
            continue
        
        # 将注意力图转换为单通道热图
        # 红色通道表示高注意力，蓝色通道表示低注意力
        red_channel = attn_img[:, :, 2].astype(float) / 255.0
        blue_channel = attn_img[:, :, 0].astype(float) / 255.0
        attn_map = red_channel - blue_channel
        attn_map = np.clip(attn_map, 0, 1)
        
       # 生成边界框（最多 5 个）
        items = generate_bboxes_from_attention(attn_map,
                                               min_area,
                                               max_boxes = 5)

        # 绘制
        result_img = draw_bboxes_on_image(orig_img, items)

        # # 保存 JPG
        output_path = os.path.join(output_dir, f"{base_name}_bbox.jpg")
        # cv2.imwrite(output_path, result_img)

        # === 新增：保存 JSON ===
        json_out = {
            "image": img_file,
            "bboxes": [
                {"box": it['box'], "score": round(it['score'],4)}
                for it in items
            ]
        }
        json_path = os.path.join(output_dir, f"{base_name}_bbox.json")
        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=2)

        print(f"Saved {len(items)} boxes → {output_path} & {json_path}")

def save_combined_visualization(orig_img, attn_img, bboxes, output_dir, base_name):
    """保存带注意力图和边界框的组合可视化"""
    # 创建组合图像
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原始图像带边界框
    orig_with_boxes = orig_img.copy()
    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(orig_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
    # BGR转RGB
    orig_with_boxes_rgb = cv2.cvtColor(orig_with_boxes, cv2.COLOR_BGR2RGB)
    attn_img_rgb = cv2.cvtColor(attn_img, cv2.COLOR_BGR2RGB)
    
    # 绘制子图
    axs[0].imshow(orig_with_boxes_rgb)
    axs[0].set_title('Original Image with Bounding Boxes')
    axs[0].axis('off')
    
    axs[1].imshow(attn_img_rgb)
    axs[1].set_title('Attention Map')
    axs[1].axis('off')
    
    # 保存组合图像
    combined_path = os.path.join(output_dir, f"{base_name}_combined.jpg")
    plt.tight_layout()
    plt.savefig(combined_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved combined visualization to: {combined_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate bounding boxes from DINO attention maps')
    parser.add_argument('--original_dir', default='/home/haoding/Wenzheng/dinov2/figures',
                        help='Directory containing original images')
    parser.add_argument('--attn_dir', default='/home/haoding/Wenzheng/dinov2/figures_attn_dino1',
                        help='Directory containing DINO attention maps')
    parser.add_argument('--output_dir', default='/home/haoding/Wenzheng/dinov2/bbox',
                        help='Output directory for images with bounding boxes')
    parser.add_argument('--min_area', type=int, default=100,
                        help='Minimum area for detected tools (default: 100)')
    parser.add_argument('--max_boxes', type=int, default=5,
                        help='Maximum number of bounding boxes to detect (default: 3)')
    
    args = parser.parse_args()
    
    # 处理所有图像
    process_images(
        original_dir=args.original_dir,
        attn_dir=args.attn_dir,
        output_dir=args.output_dir,
        min_area=args.min_area,
        max_boxes=args.max_boxes
    )

if __name__ == "__main__":
    main()