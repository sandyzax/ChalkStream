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
        detectShadows=False
    )
    mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    previous_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if previous_frame is None:
            previous_frame = frame.copy()

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

        final_image = cv2.bitwise_or(movement_filtered,cv2.bitwise_and(previous_frame,filtered_out_area))

        cv2.imshow('Original', frame)
        cv2.imshow('Professor Filtered', final_image)
        
        previous_frame = final_image

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run()