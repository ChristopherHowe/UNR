import numpy as np
import math
from project3 import pad_img, unpad_img


def old_apply_filter(image: np.ndarray, mask: np.ndarray, pad_pixels: int, pad_value: int):
    def correlation(image: np.ndarray, mask: np.ndarray, img_x: int, img_y: int):
        val: int = 0
        for mask_x in range(mask_w):
            for mask_y in range(mask_h):
                x_diff = int(mask_x - (mask_w / 2) + 0.5)
                y_diff = int(mask_y - (mask_h / 2) + 0.5)
                src_val = np.mean(image[img_x + x_diff][img_y + y_diff])
                step = src_val * mask[mask_x][mask_y]
                val += step
        return val

    def handle_mask_check(mask: np.ndarray):
        if mask.ndim == 1:  # make 1D arrays into 2D with width 1
            mask = mask.reshape(1, -1)
        if mask.ndim > 2:
            raise ValueError("Does not support masks with a higher dimension than 2")
        return mask

    mask = handle_mask_check(mask)
    mask_w = mask.shape[0]
    mask_h = mask.shape[1] if mask.ndim == 2 else 0
    req_w_space = math.floor(mask_w / 2)
    req_h_space = math.floor(mask_h / 2)

    src, pad_values = pad_img(image, pad_pixels, pad_value)
    src_w = src.shape[0]
    src_h = src.shape[1]

    new_img = np.zeros(src.shape, dtype=np.float64)
    for img_x in range(req_w_space, src_w - req_w_space):
        for img_y in range(req_h_space, src_h - req_h_space):
            v = correlation(src, mask, img_x, img_y)
            new_img[img_x][img_y] = v

    new_img = unpad_img(new_img, pad_values)
    return new_img
