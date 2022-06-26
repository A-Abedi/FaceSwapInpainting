import numpy as np
import cv2

from Segmentation import get_head


def change_solid_bw(image: np.ndarray) -> np.ndarray:
    """
    Change the 0, 0, 0 to 1, 1, 1
    Change the 255, 255, 255 to 254, 254, 254
    Args:
        image: The source image.

    Returns: The change image.
    """
    image[image == 0] = 1
    image[image == 255] = 254

    return image


def remove_background(image: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """
    Remove the image background using its SCHP segmentation.
    Args:
        image: The source image.
        segment: The segmentation of the image.
    """
    kernel = np.ones((70,70), np.uint8)
    img_dilation = cv2.dilate(segment, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    segment = img_erosion.copy()
    image[segment == 0] = 254


def remove_border(image: np.ndarray, segment: np.ndarray, border: int = 10) -> (np.ndarray, np.ndarray):
    """
    Remove the border of the body. The result is an image contains only the body.
    Args:
        image: The source image.
        segment: The segmentation result.
        border: The image border.

    Returns: The change image and segmentation result.
    """
    a, thresh = cv2.threshold(segment, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)

    return image[max(y - border, 0):min(y + h + border, image.shape[0]),
           max(x - border, 0):min(x + w + border, image.shape[1])], \
           segment[max(y - border, 0):min(y + h + border, image.shape[0]),
           max(x - border, 0):min(x + w + border, image.shape[1])], (y - (border/2))


def resize_image_by_ratio(image: np.ndarray, ratio_model: tuple, ratio_user: tuple) -> np.ndarray:
    """
    Resize the image by shoulder width and body height.
    Args:
        image: The source image.
        ratio_model: The model body ratio.
        ratio_user: The user body ratio.

    Returns: The changed-size image.
    """
    shoulder = ratio_user[0] / ratio_model[0]
    up_height = ratio_user[1] / ratio_model[1]

    print("Shoulder fx:", shoulder)
    print("Up height fx:", up_height)

    return cv2.resize(image, None, fx=shoulder, fy=up_height)



def boundary_finder(pixels):
    """
    Get a convex shape that covers the given pixels.
    Args:
        pixels: The pixels.

    Returns: The found convex hull pixels.
    """
    convexhull_pixels = cv2.convexHull(pixels)
    boundary_pixels = [[point[0][1], point[0][0]] for point in convexhull_pixels]

    return boundary_pixels


def remove_head_and_hair(image: np.ndarray, segment: np.ndarray):
    """
    Remove hair and hair in an image.
    Args:
        image: The source image.
        segment: The segmentation of the image.
    """
    hh_filter = (segment == 2) | (segment == 13)
    image[np.where(hh_filter != 0)] = [255, 255, 255]


def translate_face(model_segment: np.ndarray, user_segment: np.ndarray, model_image: np.ndarray, user_image: np.ndarray, lip_loc):
    """
    Translate the user image to a suitable place in model image using Affine translation matrix.
    Args:
        model_segment: Model segmentation.
        user_segment: User segmentation.
        model_image: Model image.
        user_image: User image.

    Returns: The translated user image and the translated user segmentation.
    """
    
    model_head = get_head(model_segment)
    user_head = get_head(user_segment)

    model_head_indices = np.argwhere(model_head == 1)
    model_head_bottom = np.array([int((np.max(model_head_indices, axis = 0))[0]), int(((np.min(model_head_indices, axis = 0))[1] + (np.max(model_head_indices, axis = 0))[1]) /2)])
    print("Model head bottom:", model_head_bottom)

    user_head_indices = np.argwhere(user_head == 1)
    user_head_bottom = np.array([int((np.max(user_head_indices, axis = 0))[0]), int(((np.min(user_head_indices, axis = 0))[1] + (np.max(user_head_indices, axis = 0))[1]) /2)])
    print("User head bottom:", user_head_bottom)

    head_diff = model_head_bottom - user_head_bottom

    t_matrix = np.float32([
        [1, 0, head_diff[1]],
        [0, 1, head_diff[0]],
    ])
    translated_user = cv2.warpAffine(user_image, t_matrix, (user_image.shape[1], user_image.shape[0]))
    translated_user_segment = cv2.warpAffine(user_segment, t_matrix, (user_segment.shape[1], user_segment.shape[0]))

    hh_filter = (translated_user_segment == 2) | (translated_user_segment == 13)

    model_image[np.where(hh_filter != 0)] = translated_user[np.where(hh_filter != 0)]
    
    translated_lip = t_matrix @ [1, lip_loc, 1]

    return translated_user, translated_user_segment, translated_lip


def to_same_size(model_image: np.ndarray, user_image: np.ndarray, model_segment: np.ndarray, user_segment: np.ndarray, w):
    """
    Zero padding the images (Based on which one is larger).
    Zero paddings are applied from top and left.
    Args:
        model_image: The model image.
        user_image: The user image.
        model_segment: The model segmentation.
        user_segment: The user segmentation.

    Returns: Model image, User image, Model segment and user segment.
    """
    model_w, model_h, _ = model_image.shape
    user_w, user_h, _ = user_image.shape

    w_diff = model_w - user_w
    h_diff = model_h - user_h

    if w_diff > 0:
        user_image = np.pad(user_image, ((w_diff, 0), (0, 0), (0, 0)), constant_values=254)
        user_segment = np.pad(user_segment, ((w_diff, 0), (0, 0)), constant_values=0)
        w -= w_diff
    elif w_diff < 0:
        model_image = np.pad(model_image, ((abs(w_diff), 0), (0, 0), (0, 0)), constant_values=254)
        model_segment = np.pad(model_segment, ((abs(w_diff), 0), (0, 0)), constant_values=0)

    if h_diff > 0:
        user_image = np.pad(user_image, ((0, 0), (h_diff, 0), (0, 0)), constant_values=254)
        user_segment = np.pad(user_segment, ((0, 0), (h_diff, 0)), constant_values=0)
    elif h_diff < 0:
        model_image = np.pad(model_image, ((0, 0), (abs(h_diff), 0), (0, 0)), constant_values=254)
        model_segment = np.pad(model_segment, ((0, 0), (abs(h_diff), 0)), constant_values=0)

    return model_image, user_image, model_segment, user_segment, w


def get_seamless_mask(user_segment):
    """
    Get the seamless cloning mask. The changes are occurred by reference.
    Args:
        user_segment: The user image segmentation result.

    Returns: The center of the seamless mask.
    """
    hhn_filter = (user_segment == 2) | (user_segment == 13) | (user_segment == 10)
    n_filter = (user_segment == 10)
    model_neck_indices = np.argwhere(n_filter == 1)

    user_segment[~hhn_filter] = 0

    model_head_indices_bottom = model_neck_indices[-1][0]
    user_segment[model_head_indices_bottom - int(model_head_indices_bottom * 0.1):] = 0
    user_segment[user_segment != 0] = 1

    # manual method:
    # head_mask = np.argwhere(user_segment == 1)
    # cX = head_mask[0][1]
    # cY = (head_mask[0][0] + head_mask[-1][0]) // 2

    M = cv2.moments(user_segment)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def get_head_bottom(segment: np.ndarray):
    """
    Get the bottom point of the image.
    Args:
        segment: Segmented image.

    Returns: The bottom point of the image.
    """
    head = get_head(segment)

    return np.argwhere(head == 1)[-1]