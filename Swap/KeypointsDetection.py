import numpy as np
import mediapipe as mp

mpPose = mp.solutions.pose

def body_keypoints(image: np.ndarray):
    """
    Get body keypoints.
    Args:
        image: The source image.

    Returns: The body keypoints.
    """
    pose = mpPose.Pose(False, 1, True, False, True, 0.5, 0.5)
    results = pose.process(image)

    lm_list = []

    h, w, c = image.shape
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            lm_list.append([int(lm.x * w), int(lm.y * h)])

    return lm_list


def get_body_ratio(key_points: list):
    """
    Return the shoulder width and the body height.
    Args:
        key_points: The body keypoints.

    Returns: A tuple. First element is shoulder width and the second element is body height.
    """
    # shoulder width, up_height
    return abs(key_points[11][0] - key_points[12][0]), abs(key_points[23][1] - key_points[11][1])


