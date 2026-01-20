import cv2
import numpy as np
from sklearn.mixture import GaussianMixture




if __name__ == "__main__":
    # Load grayscale
    img_path = "input.png"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,  # odd
        C=5
    )

    more_white = np.count_nonzero(binary) > (binary.size // 2)

    if more_white:
        binary = cv2.bitwise_not(binary)

    color = cv2.bitwise_and(img, img, mask=binary)
    inv_mask = cv2.bitwise_not(color)

    color[inv_mask == 255] = 255

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 2.2, 0, 255).astype(np.uint8)
    hsv_enhanced = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


    cv2.imshow("gray", color)
    cv2.imshow("GMM Cutoff", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

