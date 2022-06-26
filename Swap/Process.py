import cv2
import numpy as np

from .KeypointsDetection import body_keypoints, get_body_ratio
from .Utils import resize_image_by_ratio, remove_border, remove_background, to_same_size, get_seamless_mask
from Segmentation import get_skin, segment_image
from .SkinColor import change_skin_color_model
from matplotlib import pyplot as plt


def resize_image(model_image, user_image):
    """
    Resize the model image to the user image size
    Args:
        model_image: The model image
        user_image: The user image

    Returns: Resized model image.
    """
    # Get keypoints
    model_keypoints = body_keypoints(model_image)
    user_keypoints = body_keypoints(user_image)
        
    # lip finder
    lip_user = user_keypoints[9]

    # Get body ratios
    model_ratio = get_body_ratio(model_keypoints)
    print("Model ratio:", model_ratio)
    user_ratio = get_body_ratio(user_keypoints)
    print("User ratio:", user_ratio)
    

    # Resize the model image
    return resize_image_by_ratio(model_image, model_ratio, user_ratio), lip_user

def segment_preprocesses(model_image, user_image, schp_model_path):
    """
    Segment and remove background of the model and user images.
    Remove the border of the images.
    Zero padding (From top and left) the images to have the same size.
    This process significantly reduce the calculation.
    Args:
        model_image: The model image
        user_image: The user image
        schp_model_path: Path to the SCHP model.

    Returns: Model and user image. Model and User segments.
    """
    w_img, _, _ = user_image.shape
    model_segment = segment_image(model_image)
    user_segment = segment_image(user_image)

    remove_background(model_image, model_segment)
    remove_background(user_image, user_segment)

    model_image, model_segment, model_w = remove_border(model_image, model_segment)
    user_image, user_segment, user_w = remove_border(user_image, user_segment)

    return to_same_size(model_image, user_image, model_segment, user_segment, user_w)


def change_skin_color(model_image, user_image, model_segment, user_segment):
    """
    Change the model skin color to user skin color.
    Args:
        model_image: Model image
        user_image: User image
        model_segment: The segmentation of the model
        user_segment: The segmentation of the user

    Returns: The image of the model.
    """
    model_skin = get_skin(model_segment, face=False)
    print("Model skin color:", model_skin)
    user_skin = get_skin(user_segment, face=True)
    print("User skin color:", user_skin)

    return change_skin_color_model(model_image, user_image, model_skin, user_skin)


def apply_seamless_cloning(model_image, user_image, user_segment):
    """
    Apply seamless cloning on the final image.
    Args:
        model_image: The image of the model.
        user_image: The image of the user.
        user_segment: The segmentation of the user

    Returns: The result of the seamless cloning.
    """
    hhn_filter = (user_segment == 2) | (user_segment == 13) | (user_segment == 10)
    user_image[~hhn_filter] = 0

    center_x, center_y = get_seamless_mask(user_segment)
    print("User head center:", center_x, center_y)

    user_image = user_image[..., ::-1].astype("uint8")
    model_image = model_image[..., ::-1].astype("uint8")

    user_segment[user_segment == 1] = 255
    return cv2.seamlessClone(user_image, model_image, user_segment, (int(center_x), int(center_y)), cv2.NORMAL_CLONE)

def apply_inpainting(model_image, lip_user):
    # mask for pixels below lip
    mask = np.full((model_image.shape[0],model_image.shape[1]), 0)
    for i in range(int(lip_user[1]), model_image.shape[0]):
      for j in range(model_image.shape[1]):
        if list(model_image[i, j]) == [255,255,255]:
          mask[i, j] = 255

    model_image = cv2.inpaint(model_image.astype('uint8'), mask.astype('uint8'), 3, cv2.INPAINT_TELEA)
    return model_image, mask

def find_lip_loc(lip_user, w):
    first_loc = lip_user[1]
    final_lip_loc = first_loc - w
    return final_lip_loc

