import cv2
from utils.visualization_utils import generate_random_colors
import numpy as np

def downsample(image, scale_factor):
    return cv2.resize(image, (int(scale_factor * image.shape[1]), int(scale_factor * image.shape[0])), interpolation = cv2.INTER_NEAREST)

def masks_to_image(masks):
    '''
    Function that takes an array of masks and returns an pixel-wise label map
    '''
    image_labels = np.zeros(masks[0]['segmentation'].shape)
    for i, mask in enumerate(masks):
        image_labels[mask['segmentation']] = i + 1

    return image_labels

def masks_to_colored_image(masks):
    '''
    Function that takes an array of masks and returns an pixel-wise colored label map
    '''
    colors = generate_random_colors(200)
    height, width = masks[0]["segmentation"].shape
    image_labels = np.zeros((height, width, 3))
    for i, mask in enumerate(masks):
        image_labels[mask['segmentation']] = colors[i]

    return image_labels