import numpy as np
import cv2
from skimage import filters, transform

def diagram_vectorization(cv_image, vectorization=False):
    # 1. Load and Preprocess
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # 2. Binarization
    # If image is 0-255, offset should be an integer (e.g., 10)
    thresh = filters.threshold_local(gray_image, block_size=35, offset=20)
    bw = gray_image < thresh  # This creates the mask: True where there is ink
    
    # 3. Create a Single-Channel Digital Canvas (White background)
    digital_canvas = np.ones_like(gray_image) * 255

    if vectorization:
        # Identify and Straighten Lines
        # Probabilistic Hough expects a binary image (0 and 1 or 0 and 255)
        # We pass it as uint8 for compatibility
        lines = transform.probabilistic_hough_line(bw.astype(np.uint8), 
                                                 threshold=10, 
                                                 line_length=100, 
                                                 line_gap=5)

        for line in lines:
            p0, p1 = line
            # Draw perfectly straight lines on our canvas
            cv2.line(digital_canvas, p0, p1, 0, thickness=2)

    # 4. Overlay "Everything Else"
    # This places the cleaned handwriting/text back onto the canvas
    digital_canvas[bw] = 0 

    return digital_canvas

if __name__ == "__main__":
    image = cv2.imread("i3.jpg")
    if image is None:
        print("Error: Could not find image.")
    else:
        # Test with vectorization ON to see the straightening
        diagram = diagram_vectorization(image)
        
        cv2.imshow("Original", cv2.resize(image, (0,0), fx=0.5, fy=0.5))
        cv2.imshow("Digitalized", cv2.resize(diagram, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()