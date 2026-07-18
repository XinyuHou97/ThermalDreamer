#!/bin/bash
export PYTHONWARNINGS=ignore

# ========== Configuration ==========
DATA="outdoor"
CUDA_DEVICE=3
scene_prompt="campus"
control_mode="depth"  # Options: 'depth', 'canny', 'fuse'

# ========== Main Script ==========

# 1. Preprocess & thermal enhancement

echo "1. Processing data, thermal enhancement: $DATA"
python thermal_enhancement/thermal_enh.py --data "$DATA"


# 2. Depth estimation

echo "2. Depth estimation: $DATA"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python depth_estimation/run.py --data "$DATA"


# 3. Scene latent (optional)

echo "3. Scene latent processing: $DATA"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python scene_latent/firstframe.py --data "$DATA" --prompt "$scene_prompt" --control_mode "$control_mode"


# 4. Video generation (optional)

echo "4. Video generation: $DATA"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python video_generator/generate.py --data "$DATA" --control_mode "$control_mode" --prompt "both" --text_prompt "$scene_prompt" 



 