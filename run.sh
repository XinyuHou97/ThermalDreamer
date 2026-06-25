#!/bin/bash
export PYTHONWARNINGS=ignore

# ========== Configuration ==========
DATA="office"
CUDA_DEVICE=3
scene_prompt="an office with blue carpeting, featuring a row of beige lockers on the right"

# Processing stages
THERMAL_ENHANCEMENT=true
DEPTH_ESTIMATION=true
SCENE_LATENT=false
VIDEO_GENERATION=false

# ========== Main Script ==========

# 1. Preprocess & thermal enhancement
if [ "$THERMAL_ENHANCEMENT" = true ]; then
    echo "1. Processing data, thermal enhancement: $DATA"
    python thermal_enhancement/thermal_enh.py --data "$DATA"
fi

# 2. Depth estimation
if [ "$DEPTH_ESTIMATION" = true ]; then
    echo "2. Depth estimation: $DATA"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python depth_estimation/run.py --data "$DATA"
fi

# 3. Scene latent (optional)
if [ "$SCENE_LATENT" = true ]; then
    echo "3. Scene latent processing: $DATA"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python scene_latent/firstframe.py --data "$DATA" --prompt "$scene_prompt"
fi

# 4. Video generation (optional)
if [ "$VIDEO_GENERATION" = true ]; then
    echo "4. Video generation: $DATA"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python video_generator/generate.py --data "$DATA"
fi


 