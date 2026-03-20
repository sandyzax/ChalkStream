import cv2
import numpy as np
import mediapipe as mp

"""
This script remove noises 
"""

previous_frame = None
avg_background = None
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
alpha = 0.3
blend = 0.1

fgbg = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=25,
        detectShadows=True
    )
fgbg.setShadowThreshold(0)

def noise_removal(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_selfie_segmentation.process(rgb)
    mask = cv2.bitwise_not((result.segmentation_mask > 0.25).astype('uint8') * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))
    shrunk_mask = cv2.erode(mask, kernel, iterations=1)
    people_filtered = cv2.bitwise_and(frame, frame, mask=shrunk_mask)
    
    #filter out movements
    fgmask = fgbg.apply(people_filtered)
    bgmask = cv2.bitwise_not(fgmask)
    cv2.accumulateWeighted(people_filtered, avg_background, alpha, mask=shrunk_mask)
    background = cv2.convertScaleAbs(avg_background)
    output = cv2.addWeighted(noise_removal.previous_frame, 1-blend, background, blend, 0)
    movement_filtered = cv2.bitwise_and(output, output, mask=bgmask)

    #combine
    filtered_out_area = cv2.bitwise_or(fgmask,cv2.bitwise_not(shrunk_mask))
    filtered_out_area = cv2.cvtColor(filtered_out_area, cv2.COLOR_GRAY2BGR)

    final_image = cv2.bitwise_or(movement_filtered,cv2.bitwise_and(noise_removal.previous_frame,filtered_out_area))
    noise_removal.previous_frame = final_image.copy()
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

        #filter out people
        final_image = noise_removal(frame)

        cv2.imshow('Original', frame)
        cv2.imshow('Professor Filtered', final_image)
        
        

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run()