import numpy as np
import cv2

def compute_heatmap_strength(heatmap_img):
    """
    Returns a score between 0 and 1
    Higher = more concentrated disease attention
    """

    gray = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0

    # pixels with strong activation
    strong_pixels = gray > 0.3

    ratio = np.sum(strong_pixels) / gray.size
    return round(ratio, 3)
