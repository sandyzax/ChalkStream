import cv2
import numpy as np
import mediapipe as mp

"""
This script remove noises 
"""
# noise removal:
DEFAULT_PREPROCESS_HISTORY = 50
DEFAULT_PREPROCESS_VAR_THRESHOLD = 25.0
_noise_settings = {
    "history": DEFAULT_PREPROCESS_HISTORY,
    "var_threshold": DEFAULT_PREPROCESS_VAR_THRESHOLD,
    
}


def _create_background_subtractor(history, var_threshold):
    fgbg =  cv2.createBackgroundSubtractorMOG2(
        history=int(history),
        varThreshold=float(var_threshold),
        detectShadows=True,
    )
    fgbg.setShadowThreshold(0)
    return fgbg


fgbg = _create_background_subtractor(
    DEFAULT_PREPROCESS_HISTORY,
    DEFAULT_PREPROCESS_VAR_THRESHOLD,
)
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)


def _ensure_background_subtractor(settings_snapshot=None):
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


def noise_removal(frame, settings_snapshot=None):
    #filter out people
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_selfie_segmentation.process(rgb)
    mask = cv2.bitwise_not((result.segmentation_mask > 0.4).astype('uint8') * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18,18))
    shrunk_mask = cv2.erode(mask, kernel, iterations=1)
    people_filtered = cv2.bitwise_and(frame, frame, mask=shrunk_mask)

    #filter out movements
    fgmask = _ensure_background_subtractor(settings_snapshot).apply(people_filtered)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18,18))
    shrunk_mask2 = cv2.dilate(fgmask, kernel2, iterations=1)
    bgmask = cv2.bitwise_not(shrunk_mask2)
    movement_filtered = cv2.bitwise_and(people_filtered, people_filtered, mask=bgmask)

    filtered_out_area = cv2.bitwise_or(shrunk_mask2,cv2.bitwise_not(shrunk_mask))
    filtered_out_area = cv2.cvtColor(filtered_out_area, cv2.COLOR_GRAY2BGR)

    final_image = cv2.bitwise_or(movement_filtered,cv2.bitwise_and(noise_removal.previous_frame,filtered_out_area))
    noise_removal.previous_frame = final_image
    return final_image

def run():
    cap = cv2.VideoCapture("Test3.mp4")
    ret, frame = cap.read()
    if not ret:
        exit()
    
    
    noise_removal.previous_frame = frame.copy()
    global avg_background
    avg_background = frame.astype(np.float32)

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = noise_removal(frame)

        cv2.imshow('Original', frame)
        cv2.imshow('Professor Filtered', output)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run()