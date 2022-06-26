import cv2
import numpy as np


def change_skin_color_model(model, user, model_skin, user_skin):
    """
    Change the sking color of the model image.
    Args:
        model: Model image
        user: User image
        model_skin: Model skin mask
        user_skin: User skin mask

    Returns: The model image with changed skin color.
    """
    model_skin_color = _get_skin_color(model, model_skin)
    user_skin_color = _get_skin_color(user, user_skin)

    skin_color_diff = np.subtract(user_skin_color, model_skin_color)

    model = model.astype("int16")
    model[model_skin != 0] += skin_color_diff[:3].astype("int16")[::-1]

    return np.clip(model, 0, 255, out=model)


def _get_skin_color(image: np.ndarray, skin_image: np.ndarray) -> np.ndarray:
    """
    A private function to get the skin color (The mean of the skin pixels value)
    Args:
        image: The image
        skin_image: The skin mask

    Returns: The skin color.
    """
    return cv2.mean(image[..., ::-1], skin_image)


def get_skin_color(image: np.ndarray, image_segment: np.ndarray) -> np.ndarray:
    """
    Public function to get skin color
    Args:
        image: The image
        image_segment: The skin mask

    Returns: The skin color.
    """
    return _get_skin_color(image, image_segment)
