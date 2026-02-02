import cv2
import numpy as np

def estimate_severity(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    infected_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size

    percent = (infected_pixels / total_pixels) * 100

    if percent < 10:
        level = "Mild"
    elif percent < 30:
        level = "Moderate"
    else:
        level = "Severe"

    return round(percent,2), level
