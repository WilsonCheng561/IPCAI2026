import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from segments import segment_by_phases
from sam2.build_sam import build_sam2_video_predictor

def main(sam2_checkpoint="checkpoints/sam2_hiera_large.pt", 
         model_cfg="sam2_hiera_l.yaml", 
         video_dir="/mnt/disk0/haoding/cholec80/frames/video01", 
         annotation_file="/mnt/disk0/haoding/cholec80/phase_annotations/video01-phase.txt"):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # take frames from segments
    segment_dict = segment_by_phases(video_dir, annotation_file)

    # try preparation first
    frame_names = segment_dict['Preparation'][0]

    # take a look at the first video frame
    frame_idx = 0
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

    inference_state = predictor.init_state(video_path=video_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SAM 2 video predictor")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2_hiera_large.pt", help="Path to SAM 2 checkpoint")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml", help="Path to model configuration file")
    parser.add_argument("--video_dir", type=str, default="/mnt/disk0/haoding/cholec80/frames/video01", help="Path to video directory")
    parser.add_argument("--annotation_file", type=str, default="/mnt/disk0/haoding/cholec80/phase_annotations/video01-phase.txt", help="Path to annotation file")

    args = parser.parse_args()

    main(sam2_checkpoint=args.sam2_checkpoint, 
         model_cfg=args.model_cfg, 
         video_dir=args.video_dir, 
         annotation_file=args.annotation_file)