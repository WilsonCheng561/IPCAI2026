import os
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

from no_time_to_train.dataset.visualization import vis_coco, generate_distinct_colors

def load_json_data(json_path):
    """Load and parse the JSON file containing annotations."""
    with open(json_path, 'r') as f:
        return json.load(f)

def process_annotations(annotations, image_info):
    """Process annotations to extract masks, bboxes, and labels."""
    masks = []
    bboxes = []
    category_ids = []
    scores = []
    
    for ann in annotations:
        # Convert segmentation to binary mask
        if isinstance(ann['segmentation'], dict):  # RLE format
            mask = mask_utils.decode(ann['segmentation'])
        else:  # Polygon format
            height, width = image_info['height'], image_info['width']
            rles = mask_utils.frPyObjects(ann['segmentation'], height, width)
            mask = mask_utils.decode(mask_utils.merge(rles))
        
        masks.append(mask)
        bboxes.append(ann['bbox'])  # [x, y, width, height]
        category_ids.append(ann['category_id'])
        scores.append(1.0)  # For reference images, we assume perfect confidence
    
    if masks:
        masks = np.stack(masks)
        bboxes = np.array(bboxes)
        # Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    
    return np.array(masks), np.array(bboxes), np.array(category_ids), np.array(scores)

def plot_reference_images(json_path, image_dir, output_dir, dataset_name='COCO'):
    """Plot reference images with their annotations."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    data = load_json_data(json_path)
    
    # Create category_id to index mapping
    category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(data['categories'])}
    
    # Create image_id to image_info mapping
    image_info_map = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Process each image
    for image_id, annotations in annotations_by_image.items():
        image_info = image_info_map[image_id]
        img_path = os.path.join(image_dir, image_info['file_name'])
        out_path = os.path.join(output_dir, f"ref_{image_info['file_name']}")
        
        # Process annotations
        masks, bboxes, category_ids, scores = process_annotations(annotations, image_info)
        
        # Convert category IDs to indices
        labels = np.array([category_id_to_idx[cat_id] for cat_id in category_ids])
        
        # Visualize using the existing visualization function
        vis_coco(
            gt_bboxes=bboxes,
            gt_labels=labels,
            gt_masks=masks,
            scores=scores,
            labels=labels,
            bboxes=bboxes,
            masks=masks,
            score_thr=0.0,  # Show all annotations
            img_path=img_path,
            out_path=out_path,
            show_scores=False,
            dataset_name=dataset_name
        )

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot reference images with annotations')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file with annotations')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for visualizations')
    parser.add_argument('--dataset_name', type=str, default='COCO', help='Dataset name for color scheme')
    
    args = parser.parse_args()
    plot_reference_images(args.json_path, args.image_dir, args.output_dir, args.dataset_name)

if __name__ == '__main__':
    main()
