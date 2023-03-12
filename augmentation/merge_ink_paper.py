import numpy as np


def merge(img_ink: np.ndarray, img_paper: np.ndarray):
    mask = get_mask(img_ink)
    inv_mask = 1.0 - mask
    return mask * img_ink + img_paper * inv_mask


def get_mask(img: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    mask = np.full((h, w, 1), 1, dtype=np.uint8)

    white = np.all(img == [1, 1, 1], axis=-1)
    mask[white] = 0
    return mask
