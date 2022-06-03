"""
Modified from CRAFT repo, to accept batches
"""

import os
import time

import cv2
import numpy as np

import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.image_utils as image_utils
import craft_text_detector.torch_utils as torch_utils

from torch import stack

def get_prediction_batch(
    images,
    craft_net,
    refine_net=None,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    cuda: bool = False,
    long_size: int = 1280,
    poly: bool = True,
):
    """
    Arguments:
        images: list of path to the image to be processed or numpy array or PIL image
        output_dir: path to the results to be exported
        craft_net: craft net model
        refine_net: refine net model
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        canvas_size: image size for inference
        long_size: desired longest image size for inference
        poly: enable polygon type
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualizations of the detected characters/links,
         "times": elapsed times of the sub modules, in seconds}
    """
    t0 = time.time()

    # read/convert image
    batch_size = len(images)
    x_batch = []
    for image in images:
        image = image_utils.read_image(image)

        # resize
        img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(
            image, long_size, interpolation=cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio
        #resize_time = time.time() - t0
        #t0 = time.time()

        # preprocessing
        x = image_utils.normalizeMeanVariance(img_resized)
        x = torch_utils.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x_batch.append(x)
        
    x = torch_utils.Variable(stack(x_batch, dim=0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()
    preprocessing_time = time.time() - t0
    t0 = time.time()

    # forward pass
    with torch_utils.no_grad():
        y, feature = craft_net(x)
    craftnet_time = time.time() - t0
    t0 = time.time()

    # make score and link map
    score_texts = y[:, :, :, 0].cpu().data.numpy()
    #score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch_utils.no_grad():
            y_refiner = refine_net(y, feature)
        score_links = y_refiner[:, :, :, 0].cpu().data.numpy()
    refinenet_time = time.time() - t0
    t0 = time.time()

    # Post-processing
    boxess = []
    for i in range(batch_size):
        score_text = score_texts[i]
        score_link = score_links[i]
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly
        )

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        
        boxess.append(boxes)

    postprocess_time = time.time() - t0

    times = {
        "preprocessing_time": preprocessing_time,
        "craftnet_time": craftnet_time,
        "refinenet_time": refinenet_time,
        "postprocess_time": postprocess_time,
    }

    return {
        "boxess": boxess,
        "times": times,
    }
