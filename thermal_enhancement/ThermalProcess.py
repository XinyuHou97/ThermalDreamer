import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from fieldscale import Fieldscale
import os 

def load_and_normalize_images(folder_path):
    """
    Load thermal images from a folder and normalize all images between the global min and max.

    Args:
        folder_path (str): Path to the folder containing thermal images.

    Returns:
        dict: A dictionary with file names as keys and normalized images as values.
    """
    # Get a list of image files in the folder
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])

    # Initialize lists to store image data and file names
    images = []
    names = []
    # Load images and store them in a list
    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        # Load image as a grayscale image
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
            names.append(file_name)

    # Concatenate all images to compute the global min and max
    all_pixels = np.concatenate([img.flatten() for img in images])
    global_min = all_pixels.min()
    global_max = all_pixels.max()

    # Normalize images and store them in a dictionary
    normalized_images = []
    for img in images:
        normalized_img = (img - global_min) / (global_max - global_min)
        normalized_images.append(normalized_img)

    return names,normalized_images

def load_and_normalize_images_obv(image_files):
    """
    Load thermal images from a folder and normalize all images between the global min and max.

    Args:
        folder_path (str): Path to the folder containing thermal images.

    Returns:
        dict: A dictionary with file names as keys and normalized images as values.
    """
    # # Get a list of image files in the folder
    # image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])

    # Initialize lists to store image data and file names
    images = []
    # names = []
    # Load images and store them in a list
    for file_path in image_files:

        # Load image as a grayscale image
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
            # names.append(file_name)

    # Concatenate all images to compute the global min and max
    all_pixels = np.concatenate([img.flatten() for img in images])
    global_min = all_pixels.min()
    global_max = all_pixels.max()

    # Normalize images and store them in a dictionary
    normalized_images = []
    for img in images:
        normalized_img = (img - global_min) / (global_max - global_min)
        normalized_images.append(normalized_img)

    return normalized_images



def ThermalProcess_minmax(image, tmin=-1, tmax=-1):
    if tmin==-1:
        tmin = image.min()
    if tmax==-1:
        tmax = image.max()
    
    return (image-tmin)/(tmax-tmin)

def ThermalProcess_rearrange(images, bin_num=30):
    rearrange = []

    tmp_img = np.concatenate([im.flatten() for im in images])
    hist, bin_edges = np.histogram(tmp_img, bins=bin_num, range=(tmp_img.min(), tmp_img.max()))
    imgs_max = tmp_img.max()
    imgs_min = tmp_img.min()
    itv = (imgs_max - imgs_min) / bin_num
    total_num = hist.sum()

    for im in images:  # HW format
        H, W = im.shape
        mul_mask_ = np.zeros((bin_num, H, W))
        sub_mask_ = np.zeros((bin_num, H, W))
        subhist_new_min = imgs_min

        for x in range(bin_num):
            subhist = (im > imgs_min + itv * x) & (im <= imgs_min + itv * (x + 1))
            if subhist.sum() == 0:
                continue

            subhist_new_itv = hist[x] / total_num
            mul_mask_[x, ...] = subhist * (subhist_new_itv / itv)
            sub_mask_[x, ...] = subhist * (subhist_new_itv / itv * -(imgs_min + itv * x) + subhist_new_min)
            subhist_new_min += subhist_new_itv

        mul_mask = mul_mask_.sum(axis=0, keepdims=False)
        sub_mask = sub_mask_.sum(axis=0, keepdims=False)
        im_ = mul_mask * im + sub_mask
        rearrange.append(im_)
    
    return rearrange


def ThermalProcess_CLAHE(images, CLAHE_clip=3, CLAHE_tilesize=8):
    imgs = []
    CLAHE = cv2.createCLAHE(clipLimit=CLAHE_clip, tileGridSize=(CLAHE_tilesize, CLAHE_tilesize))
    for im in images:
        im = CLAHE.apply((im * 255).astype(np.uint8)).astype(np.float32)
        img_out = im / 255.0  # Normalize back to [0, 1]
        imgs.append(img_out)
    
    return imgs


# def ThermalProcess_wavelet_denoise(image, wavelet='sym4', level=3, thresh=0.6):
    



def ThermalProcess_fieldscale(image, diff=10, iteration=7, gamma=1.5, grid_size=8, clahe_clip=3, clahe=True, video=False):
    params = {
        'max_diff': diff,
        'min_diff': diff,
        'iteration': iteration,
        'gamma': gamma,
        'grid_size': grid_size,
        'clahe_clip' : clahe_clip,
        'clahe': clahe,
        'video': video
    }

    fieldscale = Fieldscale(**params)
    rescaled = fieldscale(image)
    
    return rescaled/255


