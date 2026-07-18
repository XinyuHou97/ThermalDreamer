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
    global_min = None
    global_max = None

    # Load images and track global min/max without creating a huge concatenated array
    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        # Load image as a grayscale image
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
            names.append(file_name)

            img_min = float(img.min())
            img_max = float(img.max())
            global_min = img_min if global_min is None else min(global_min, img_min)
            global_max = img_max if global_max is None else max(global_max, img_max)

    if not images:
        return names, []

    denom = global_max - global_min
    if denom == 0:
        normalized_images = [np.zeros_like(img, dtype=np.float32) for img in images]
        return names, normalized_images

    # Normalize images and store them in a dictionary
    normalized_images = []
    for img in images:
        normalized_img = (img - global_min) / denom
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

    if not images:
        return rearrange

    imgs_min = min(float(im.min()) for im in images)
    imgs_max = max(float(im.max()) for im in images)
    if imgs_max == imgs_min:
        return [im.copy() for im in images]

    hist = np.zeros(bin_num, dtype=np.int64)
    for im in images:
        hist += np.histogram(im, bins=bin_num, range=(imgs_min, imgs_max))[0]

    itv = (imgs_max - imgs_min) / bin_num
    total_num = hist.sum()

    if total_num == 0 or itv == 0:
        return [im.copy() for im in images]

    # Precompute affine mapping y = a*x + b for each bin.
    probs = hist.astype(np.float64) / float(total_num)
    a = probs / itv
    cdf_prev = np.concatenate(([0.0], np.cumsum(probs[:-1])))
    left_edges = imgs_min + itv * np.arange(bin_num, dtype=np.float64)
    b = (-left_edges * a) + cdf_prev

    upper_edges = imgs_min + itv * np.arange(1, bin_num + 1, dtype=np.float64)

    for im in images:  # HW format
        flat = im.reshape(-1)
        bin_idx = np.searchsorted(upper_edges, flat, side='left')
        bin_idx = np.clip(bin_idx, 0, bin_num - 1)

        # Match original interval behavior: (imgs_min + itv*x, imgs_min + itv*(x+1)]
        valid = flat > imgs_min
        out = np.zeros_like(flat, dtype=np.float64)
        out[valid] = a[bin_idx[valid]] * flat[valid] + b[bin_idx[valid]]
        im_ = out.reshape(im.shape)
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


