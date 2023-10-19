import cv2

def downsample(image, scale_factor):
    return cv2.resize(image, (int(scale_factor * image.shape[1]), int(scale_factor * image.shape[0])), interpolation = cv2.INTER_NEAREST)