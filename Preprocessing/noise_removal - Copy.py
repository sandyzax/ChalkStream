import cv2
import numpy as np
import mediapipe as mp

"""
This script remove noises 
"""
previous_frame = None
avg_background = None
alpha = 0.3
blend = 0.1

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, 
    varThreshold=16, # Lower threshold is more sensitive to subtle movement
    detectShadows=True
)
# Optional: Adjust shadow value (default is 127) or sensitivity
# fgbg.setShadowValue(0) # If you want shadows to just be blacked out in the mask
fgbg.setShadowThreshold(0) # 0.5 is default; lower = more shadows detected
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

def noise_removal(frame):
    global previous_frame
    global avg_background
    
    # 1. MediaPipe Selfie Segmentation
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_selfie_segmentation.process(rgb)
    
    # Create mask where people ARE (255 = person)
    person_mask = (result.segmentation_mask > 0.25).astype('uint8') * 255
    
    # Morphological expansion to ensure we cover the edges of people/clothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    person_mask = cv2.dilate(person_mask, kernel, iterations=2)
    
    # 2. Background Subtraction with Shadow Detection
    # people_filtered isolates the "static" parts of the frame for MOG2
    mask_inv = cv2.bitwise_not(person_mask)
    fgmask = fgbg.apply(frame) # Apply to whole frame to track movement correctly

    # THRESHOLD the fgmask: 
    # We want to treat real movement (255) as foreground, 
    # but we can treat shadows (127) as background if we want to update them.
    _, motion_only = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    
    # 3. Combine Masks
    # We don't want to update the background where there is a PERSON or MOTION
    combined_exclusion_mask = cv2.bitwise_or(person_mask, motion_only)
    update_mask = cv2.bitwise_not(combined_exclusion_mask)

    # 4. Weighted Accumulation
    # Only update the background model in areas with no people and no motion
    cv2.accumulateWeighted(frame, avg_background, alpha, mask=update_mask)
    background = cv2.convertScaleAbs(avg_background)

    # 5. Final Output Generation
    # If it's a "static" area, show the accumulated background.
    # If it's a "moving" area, show the previous frame to keep it "frozen" 
    # until the movement stops and it's integrated into the background.
    final_image = np.where(combined_exclusion_mask[:, :, None] == 0, background, previous_frame)
    
    previous_frame = final_image.copy()
    return final_image

def run():
    cap = cv2.VideoCapture("Test3.mp4")
    ret, frame = cap.read()
    if not ret:
        exit()
    
    global previous_frame
    previous_frame = frame.copy()
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