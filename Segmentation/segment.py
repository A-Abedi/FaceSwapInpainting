from Segmentation import schp_segment
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import ImageFilter


def _apply_transform(image: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    return transform(image)


def _apply_segmentation(input_image, schp_model_path, dataset="lip"):
    image_segments = schp_segment(input_image, schp_model_path, dataset=dataset)
    return image_segments


def _get_head_and_hair(segment: np.ndarray) -> np.ndarray:
    segment[segment == 2] = 1
    segment[segment == 13] = 1
    segment[segment != 1] = 0

    return segment


def _get_body(segment: np.ndarray) -> np.ndarray:
    segment[segment == 2] = 12
    segment[segment == 13] = 12

    return segment


def get_head(input_image, schp_model_path, dataset="lip"):
    segment = _apply_segmentation(input_image, schp_model_path, dataset)
    segment = segment.filter(ImageFilter.ModeFilter(size=13))
    segment = _get_head_and_hair(np.array(segment))

    return segment


def get_body(input_image, schp_model_path, dataset="lip"):
    segment = _apply_segmentation(input_image, schp_model_path, dataset)
    segment = segment.filter(ImageFilter.ModeFilter(size=13))
    segment = _get_body(np.array(segment))

    return segment
