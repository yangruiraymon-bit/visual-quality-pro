import cv2
import numpy as np
from rembg import remove

def get_subject_mask_rembg(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = remove(img_rgb, alpha_matting=True)
    mask = result[:, :, 3]
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask