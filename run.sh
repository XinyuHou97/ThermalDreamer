#!/bin/bash
export PYTHONWARNINGS=ignore

DATA="office"


# 1. Preprocess & thermal enhancement
echo "1. Processing data, thermal enhancement: $DATA"
python thermal_enhancement/thermal_enh.py --data "$DATA"


# 2. Depth estimation
echo "2. Depth estimation: $DATA"
CUDA_VISIBLE_DEVICES=3 python depth_estimation/run.py --data "$DATA"


