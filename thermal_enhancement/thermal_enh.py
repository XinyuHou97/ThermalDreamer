#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:14:47 2025

@author: xinyu
"""

import os
from ThermalProcess import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

thermal_folder='./data/'+args.data+'/thermal/'
savefolder= './data/'+args.data+'/thermal-enh/'

if not os.path.exists(savefolder):
    os.makedirs(savefolder)
    

_,norm_imgs=load_and_normalize_images(thermal_folder)


imgs = ThermalProcess_rearrange(norm_imgs)
imgs = ThermalProcess_CLAHE(imgs)

for idx, img in enumerate(imgs):
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(savefolder, f"{idx:06d}.png"), img)


 