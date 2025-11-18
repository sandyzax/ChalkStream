import cv2
import mediapipe as mp
import numpy as np

#Pre-Initialized variable
cap = cv2.VideoCapture("Test2.mp4")

# placeholder for now. Disable auto lighting adjustment. Require Logitech C920/C922 camera
""" 
current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
"""

# board resize:


def board_resize(frame):
    return cv2.warpPerspective(frame, board_resize.M, (board_resize.maxWidth, board_resize.maxHeight))


# noise removal:
fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=False
    )
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

def noise_removal(frame):
    #filter out people
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_selfie_segmentation.process(rgb)
    mask = cv2.bitwise_not((result.segmentation_mask > 0.25).astype('uint8') * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))
    shrunk_mask = cv2.erode(mask, kernel, iterations=1)
    people_filtered = cv2.bitwise_and(frame, frame, mask=shrunk_mask)

    #filter out movements
    fgmask = fgbg.apply(people_filtered)
    bgmask = cv2.bitwise_not(fgmask)
    movement_filtered = cv2.bitwise_and(people_filtered, people_filtered, mask=bgmask)

    filtered_out_area = cv2.bitwise_or(fgmask,cv2.bitwise_not(shrunk_mask))
    filtered_out_area = cv2.cvtColor(filtered_out_area, cv2.COLOR_GRAY2BGR)

    final_image = cv2.bitwise_or(movement_filtered,cv2.bitwise_and(noise_removal.previous_frame,filtered_out_area))
    noise_removal.previous_frame = final_image
    return final_image

# grey scale
def grey_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def run(pts):
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
        # greyed = grey_scale(noise_filtered)
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

    
