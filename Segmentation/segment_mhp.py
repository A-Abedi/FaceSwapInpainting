from Segmentation import schp_mhp_segment
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import ImageFilter


def _apply_transform(image: np.ndarray) -> torch.Tensor:
    """
    A private function to apply transformation before applying the SCHP.
    Args:
        image: The source image.

    Returns: The transformation instance.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    return transform(image)


def _apply_segmentation(input_image):
    """
    A private function to apply segmentation.
    Args:
        input_image: The source image.

    Returns: The segmentation result.
    """
    image_segments = schp_mhp_segment(input_image)
    return image_segments


def _get_head_and_hair(segment: np.ndarray) -> np.ndarray:
    """
    A private function to get hair and head from the segmentation.
    Args:
        segment: The segmentation result.

    Returns: The segmentation contains only head and hair.
    """
    segment[segment == 2] = 1
    segment[segment == 13] = 1
    segment[segment != 1] = 0

    return segment


def _get_skin(segment: np.ndarray, face: bool = False) -> np.ndarray:
    """
    Get skins of the segmentation.
    Args:
        segment: The segmentation result.
        face: Whether consider face or not.

    Returns: The segmented result.
    """
    segment[segment == 10] = 1
    segment[segment == 14] = 1
    segment[segment == 15] = 1

    if face:
        segment[segment == 13] = 1

    segment[segment != 1] = 0
    return segment


def _background_mask(segment: np.ndarray) -> np.ndarray:
    """
    Get the background of the segmentation.
    Args:
        segment: The segmentation result.

    Returns: The segmentation contains only the background.
    """
    segment[segment != 0] = 1
    return ~segment.astype(bool)


def get_head_hair(segment: np.ndarray) -> np.ndarray:
    """
    Public function to get head and hair.
    Args:
        segment: The segmentation of the image.

    Returns: The head and hair.
    """
    segment = _get_head_and_hair(segment)

    return segment


def get_head(segment: np.ndarray) -> np.ndarray:
    """
    Public function to get head mask.
    Args:
        segment:

    Returns:

    """
    return segment == 13


def get_skin(segment: np.ndarray, face: bool = False) -> np.ndarray:
    """
    Get skin mask.
    Args:
        segment: Segmentation result.
        face: Whether consider face or not.

    Returns: The skin mask.
    """
    filter_skin = (segment == 10) | (segment == 14) | (segment == 15)

    if face:
        filter_skin = filter_skin | (segment == 13)

    return np.where(filter_skin, segment, 0)


def get_neck(segment: np.ndarray) -> np.ndarray:
    """
    Public function to get the neck segmentation.
    Args:
        segment: The segmentation image.

    Returns: The neck segmentation image.
    """
    segment[segment == 10] = 1
    segment[segment != 1] = 0

    return segment


def background_mask(segment: np.ndarray) -> np.ndarray:
    """
    The background mask.
    Args:
        segment: The segmentation image.

    Returns: The background mask.
    """
    segment = _background_mask(segment)

    return segment


def segment_image(input_image):
    """
    Segment the image.
    Args:
        input_image: The input image.

    Returns: Apply image segmentation.
    """
    segment = _apply_segmentation(input_image)
    return np.array(segment.filter(ImageFilter.ModeFilter(size=13)))
