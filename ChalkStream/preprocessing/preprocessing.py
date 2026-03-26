"""Prototype preprocessing helpers used during board rectification and filtering."""

import cv2
import mediapipe as mp
import numpy as np
from core import RuntimeSettingsSnapshot, load_app_settings


PREPROCESSING_SETTINGS = load_app_settings().get("preprocessing", {})

# This module still keeps a simple demo capture path for standalone experiments.
cap = cv2.VideoCapture("Test2.mp4")

""" 
current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
"""

def board_resize(frame):
    """Apply the currently configured board warp matrix to a frame."""
    return cv2.warpPerspective(frame, board_resize.M, (board_resize.maxWidth, board_resize.maxHeight))


DEFAULT_PREPROCESS_HISTORY = int(PREPROCESSING_SETTINGS.get("history", 50))
DEFAULT_PREPROCESS_VAR_THRESHOLD = float(PREPROCESSING_SETTINGS.get("var_threshold", 25.0))
PERSON_MASK_THRESHOLD = float(PREPROCESSING_SETTINGS.get("person_mask_threshold", 0.4))
PERSON_MASK_EROSION_KERNEL = max(1, int(PREPROCESSING_SETTINGS.get("person_mask_erosion_kernel", 18)))
MOTION_MASK_DILATION_KERNEL = max(1, int(PREPROCESSING_SETTINGS.get("motion_mask_dilation_kernel", 18)))
_noise_settings = {
    "history": DEFAULT_PREPROCESS_HISTORY,
    "var_threshold": DEFAULT_PREPROCESS_VAR_THRESHOLD,
}

previous_frame = None
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
alpha = 0.3
blend = 0.1


def _create_background_subtractor(history, var_threshold):
    """Create the background subtractor used to suppress transient motion."""
    return cv2.createBackgroundSubtractorMOG2(
        history=int(history),
        varThreshold=float(var_threshold),
        detectShadows=False,
    )


fgbg = _create_background_subtractor(
    DEFAULT_PREPROCESS_HISTORY,
    DEFAULT_PREPROCESS_VAR_THRESHOLD,
)
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)


def _ensure_background_subtractor(settings_snapshot=None):
    """Rebuild the subtractor only when the live settings actually change."""
    global fgbg

    if settings_snapshot is None:
        history = DEFAULT_PREPROCESS_HISTORY
        var_threshold = DEFAULT_PREPROCESS_VAR_THRESHOLD
    else:
        history = int(getattr(settings_snapshot, "preprocess_history", DEFAULT_PREPROCESS_HISTORY))
        var_threshold = float(getattr(settings_snapshot, "preprocess_var_threshold", DEFAULT_PREPROCESS_VAR_THRESHOLD))

    if (
        fgbg is None
        or _noise_settings["history"] != history
        or abs(_noise_settings["var_threshold"] - var_threshold) > 1e-9
    ):
        fgbg = _create_background_subtractor(history, var_threshold)
        _noise_settings["history"] = history
        _noise_settings["var_threshold"] = var_threshold

    return fgbg


# def noise_removal(frame, settings_snapshot=None):
#     #filter out people
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = mp_selfie_segmentation.process(rgb)
#     mask = cv2.bitwise_not((result.segmentation_mask > 0.4).astype('uint8') * 255)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18,18))
#     shrunk_mask = cv2.erode(mask, kernel, iterations=1)
#     people_filtered = cv2.bitwise_and(frame, frame, mask=shrunk_mask)

#     #filter out movements
#     fgmask = _ensure_background_subtractor(settings_snapshot).apply(people_filtered)
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18,18))
#     shrunk_mask2 = cv2.dilate(fgmask, kernel2, iterations=1)
#     bgmask = cv2.bitwise_not(shrunk_mask2)
#     movement_filtered = cv2.bitwise_and(people_filtered, people_filtered, mask=bgmask)

#     filtered_out_area = cv2.bitwise_or(shrunk_mask2,cv2.bitwise_not(shrunk_mask))
#     filtered_out_area = cv2.cvtColor(filtered_out_area, cv2.COLOR_GRAY2BGR)

#     final_image = cv2.bitwise_or(movement_filtered,cv2.bitwise_and(noise_removal.previous_frame,filtered_out_area))
#     noise_removal.previous_frame = final_image
#     return final_image
def noise_removal(frame, settings_snapshot=None):
    """Remove people and motion while keeping the most recent stable board state."""
    # Mask out people first so they do not pollute the board image.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_selfie_segmentation.process(rgb)
    mask = cv2.bitwise_not((result.segmentation_mask > PERSON_MASK_THRESHOLD).astype('uint8') * 255)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (PERSON_MASK_EROSION_KERNEL, PERSON_MASK_EROSION_KERNEL),
    )
    shrunk_mask = cv2.erode(mask, kernel, iterations=1)
    people_filtered = cv2.bitwise_and(frame, frame, mask=shrunk_mask)

    # Then suppress new motion with background subtraction.
    fgmask = _ensure_background_subtractor(settings_snapshot).apply(people_filtered)
    kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MOTION_MASK_DILATION_KERNEL, MOTION_MASK_DILATION_KERNEL),
    )
    shrunk_mask2 = cv2.dilate(fgmask, kernel2, iterations=1)
    bgmask = cv2.bitwise_not(shrunk_mask2)
    movement_filtered = cv2.bitwise_and(people_filtered, people_filtered, mask=bgmask)

    #combine
    filtered_out_area = cv2.bitwise_or(fgmask,cv2.bitwise_not(shrunk_mask))
    filtered_out_area = cv2.cvtColor(filtered_out_area, cv2.COLOR_GRAY2BGR)

    final_image = cv2.bitwise_or(movement_filtered,cv2.bitwise_and(noise_removal.previous_frame,filtered_out_area))
    noise_removal.previous_frame = final_image.copy()
    return final_image

def grey_scale(frame):
    """Convert a BGR frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def run(pts):
    """Standalone demo entrypoint for the older preprocessing experiment."""
    widthA = np.linalg.norm(pts[3] - pts[2])
    widthB = np.linalg.norm(pts[1] - pts[0])
    board_resize.maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(pts[2] - pts[1])
    heightB = np.linalg.norm(pts[3] - pts[0])
    board_resize.maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [board_resize.maxWidth, 0],
        [board_resize.maxWidth, board_resize.maxHeight],
        [0, board_resize.maxHeight]
    ], dtype="float32")

    board_resize.M = cv2.getPerspectiveTransform(pts, dst)
    
    ret, frame = cap.read()
    noise_removal.previous_frame = board_resize(frame)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cropped = board_resize(frame)
        noise_filtered = noise_removal(cropped)
        cv2.imshow('Original', frame)
        cv2.imshow('Professor Filtered', noise_filtered)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    pts = np.array([
        [101, 86],  # top-left      [0]
        [538, 87],  # top-right     [1]
        [539,297],  # bottom-right  [2]
        [103,303]   # bottom-left   [3]
    ], dtype="float32")
    run(pts)
