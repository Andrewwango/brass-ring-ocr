"""
Functions for pre and post-processing ring images for OCR
"""
import shutil, os, math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from craft_text_detector.file_utils import rectify_poly

def moving_average_filter(arr, n=3) :
    """Smooth 1D signal arr with moving average filter of size n (default 3)
    """
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def adjust_rect(rect):
    """Adjust warped ring image so that text isn't cut off, by finding best cutoff point
    in the image.

    Args:
        rect (np.ndarray): ring image warped as a rectangle

    Returns:
        np.ndarray: adjusted rectangle
    """
    # Binarise image
    blurred = cv2.medianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), 13)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51,10)

    # Smooth image
    smoothed = moving_average_filter(thresholded.sum(axis=0), n=113)

    # Shift image
    optimum_shift = smoothed.argmax()
    return np.roll(rect, axis=1, shift=-optimum_shift)

def images_to_rects(image_filenames, foldername="test_images", irf=0.5, orf=0.5, downsample=False, plot=False, adjust=True):
    """Read batch of images and warp images from ring shape to rectangle shape.
    Note that parameters irf and orf can be directly calculated from Hough circle transform output.

    Args:
        image_filenames (list): list of test image file names.
        foldername (str, optional): folder where images reside. Defaults to "test_images".
        irf (float, optional): inner radius factor: ratio of inner ring radius to outer ring radius. Defaults to 0.5.
        orf (float, optional): outer radius factor: ratio of outer ring radius to image length. Defaults to 0.5.
        downsample (bool, optional): Whether to downsample image to save time and space. Defaults to False.
        plot (bool, optional): Whether to plot output images. Defaults to False.
        adjust (bool, optional): Whether to adjust images so text isn't cut off. Defaults to True.

    Returns:
        list: list of warped images
    """
    out_rects = []

    for fn in tqdm(image_filenames):
        ring = cv2.imread(os.path.join(foldername, fn))
        
        size = ring.shape[0]
        outer_radius = int(size * orf)

        warped = cv2.warpPolar(ring, (size, int(size * math.pi)), (outer_radius, outer_radius), outer_radius, 0)
        straightened = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cropped = straightened[: int(straightened.shape[0] * (1 - irf)), :]

        image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        if downsample:
            image = image[::4,::4,:]

        if adjust:
            image = adjust_rect(image)

        out_rects.append(image)
    
    if plot:
        for rect in out_rects:
            plt.imshow(rect)
            plt.show()
            
    return out_rects

def save_crops(rects, prediction_result):
    """Use CRAFT output bounding boxes to crop image to ROIs, and save

    Args:
        rects (list): list or batch of images
        prediction_result (dict): CRAFT output dictionary
    """
    # Empty directory
    shutil.rmtree('temp_crops')
    os.mkdir(os.path.join(os.getcwd(), 'temp_crops'))
    
    for i,boxes in enumerate(prediction_result["boxess"]):
        for j,region in enumerate(boxes):
            crop = rectify_poly(rects[i], region)
            plt.imsave(f"temp_crops/crop_{i}_{j}.png", crop)
            plt.imsave(f"temp_crops/crop_{i}_{j}_r.png", np.flip(crop, axis=(0,1)))