import numpy as np
import torch

from Segmentation.Utils import get_affine_transform
import cv2


def _box2cs(aspect_ratio, box):
    x, y, w, h = box[:4]
    return _xywh2cs(aspect_ratio, x, y, w, h)


def _xywh2cs(aspect_ratio, x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


def process(image: torch.Tensor, transform, input_size: tuple = (473, 473)):
    h, w, _ = image.shape
    aspect_ratio = input_size[1] * 1.0 / input_size[0]

    # Get person center and scale
    person_center, s = _box2cs(aspect_ratio, [0, 0, w - 1, h - 1])
    r = 0
    trans = get_affine_transform(person_center, s, r, np.asarray(input_size))
    input = cv2.warpAffine(
        image,
        trans,
        (int(input_size[1]), int(input_size[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))

    input_data = transform(input)

    meta = {
        'center': person_center,
        'height': h,
        'width': w,
        'scale': s,
        'rotation': r
    }

    return input_data, meta
