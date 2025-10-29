import cv2
import mediapipe as mp

"""
This script remove noises 
"""
def run():
    cap = cv2.VideoCapture("Test2.mp4")
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, 
        varThreshold=25,    
        detectShadows=True  
    )
    mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #filter out people
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_selfie_segmentation.process(rgb)
        mask = cv2.bitwise_not((result.segmentation_mask > 0.3).astype('uint8') * 255)
        filtered = cv2.bitwise_and(frame, frame, mask=mask)
        #filter out movements
        fgmask = fgbg.apply(filtered)
        bgmask = cv2.bitwise_not(fgmask)
        filtered = cv2.bitwise_and(filtered, filtered, mask=bgmask)

        cv2.imshow('Original', frame)
        cv2.imshow('Professor Filtered', filtered)
        

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run()