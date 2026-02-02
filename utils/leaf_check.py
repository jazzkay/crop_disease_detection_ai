import cv2
import numpy as np

def is_leaf_present(image, threshold=0.15):
    """
    Returns True if enough green pixels are present.
    """

    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = cv2.countNonZero(mask) / mask.size

    return green_ratio > threshold
