import numpy as np
import cv2
from skimage import filters, transform

def diagram_vectorization(cv_image, vectorization=False):
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    thresh = filters.threshold_local(gray_image, block_size=35, offset=20)
    bw = gray_image < thresh  
    digital_canvas = np.ones_like(gray_image) * 255

    if vectorization:
        lines = transform.probabilistic_hough_line(bw.astype(np.uint8), 
                                                 threshold=20, 
                                                 line_length=100, 
                                                 line_gap=2)

        for line in lines:
            p0, p1 = line
            cv2.line(digital_canvas, p0, p1, 0, thickness=2)

    digital_canvas[bw] = 0 
    return digital_canvas

if __name__ == "__main__":
    image = cv2.imread("i1.jpg")
    if image is None:
        print("Error: Could not find image.")
    else:
        diagram = diagram_vectorization(image,True)
        
        cv2.imshow("Original", image)
        cv2.imshow("Digitalized", diagram)
        cv2.waitKey(0)
        cv2.destroyAllWindows()