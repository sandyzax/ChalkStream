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
        history=150,
        varThreshold=12,
        detectShadows=False
    )
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

def noise_removal(frame):
    """Remove professor + movement using Mediapipe + improved masks + fill-in."""

    h, w = frame.shape[:2]

    #filter out people
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_selfie_segmentation.process(rgb)

    # Raw person mask (1 = person, 0 = background)
    mp_mask = (result.segmentation_mask > 0.25).astype('uint8') * 255

    k = int(h * 0.04)  # Kernel ≈ 4% of board height
    if k % 2 == 0:
        k += 1

    person_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    person_mask = cv2.dilate(mp_mask, person_kernel, iterations=1)

    # Remove the person from the frame
    no_person = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(person_mask))

    # 3. Motion Filtering (IMPROVED MOG2)
    fgmask = fgbg.apply(no_person)            # moving regions = white
    bgmask = cv2.bitwise_not(fgmask)          # stable regions = white

    # Remove only moving pixels
    static_only = cv2.bitwise_and(no_person, no_person, mask=bgmask)

    # 4. Removal mask for inpainting
    #    Areas to be filled:
    #       - person_mask (person is standing here)
    #       - fgmask (movement here)
    removal_mask = cv2.bitwise_or(person_mask, fgmask)
    removal_mask = cv2.cvtColor(removal_mask, cv2.COLOR_GRAY2BGR)

    # Invert to get clean, stable regions
    clean_mask = cv2.bitwise_not(removal_mask)

    # 5. Intpainting (prev frame)
    # Initialize previous frame if first call
    if not hasattr(noise_removal, "previous_frame") or noise_removal.previous_frame is None:
        noise_removal.previous_frame = frame.copy()

    clean_area = cv2.bitwise_and(static_only, clean_mask)
    restored_area = cv2.bitwise_and(noise_removal.previous_frame, removal_mask)

    final_image = cv2.bitwise_or(clean_area, restored_area)

    # Save for next frame
    noise_removal.previous_frame = final_image.copy()

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

    
