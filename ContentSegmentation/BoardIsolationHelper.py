import cv2
import numpy as np

IMAGE_PATH = "test_frames/00_23_40_00003.png"

def normalize_contrast(gray):
    # Boost local contrast to reduce shadows and glare
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def get_board_mask(bgr):
    # Convert to gray and normalize contrast
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_contrast(gray)

    # Light blur to suppress speckle without killing edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold handles uneven lighting; invert so board lines become white
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )

    # Close gaps between stroke fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Optional cleanup of tiny dots
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)

    return cleaned

def order_corners(pts4):
    # TL, TR, BR, BL
    s = pts4.sum(axis=1)
    diff = np.diff(pts4, axis=1)
    ordered = np.zeros((4,2), dtype=np.float32)
    ordered[0] = pts4[np.argmin(s)]
    ordered[2] = pts4[np.argmax(s)]
    ordered[1] = pts4[np.argmin(diff)]
    ordered[3] = pts4[np.argmax(diff)]
    return ordered

def find_board_corners(bgr, mask):
    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    h, w = mask.shape
    img_area = h * w

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.05 * img_area:  # skip tiny regions
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 1:
            continue

        # Solidity close to 1 means compact shape
        solidity = area / hull_area

        # Fit minimum-area rectangle to get a clean box
        rect = cv2.minAreaRect(hull)  # (center, (w,h), angle)
        box = cv2.boxPoints(rect)     # 4 corners float32
        box = np.array(box, dtype=np.float32)

        # Rectangularity: how much of the fitted rectangle is filled by the contour
        rect_w, rect_h = rect[1]
        rect_area = max(rect_w * rect_h, 1.0)
        rectangularity = area / rect_area

        # Aspect ratio sanity check (boards are not super skinny)
        ar = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1.0)

        # Score prefers big, solid, rectangular shapes with reasonable aspect ratio
        score = (area / img_area) + 0.8 * solidity + 0.8 * rectangularity - 0.05 * abs(ar - 1.8)

        if score > best_score:
            best_score = score
            best = (cnt, box)

    if best is None:
        return None, None

    cnt, box = best
    return cnt, order_corners(box)

def warp_board(bgr, corners, border=8):
    # Compute target size with a small extra border for context
    tl, tr, br, bl = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom)) + border * 2

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right)) + border * 2

    dst = np.array([
        [border, border],
        [max_width - border - 1, border],
        [max_width - border - 1, max_height - border - 1],
        [border, max_height - border - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, M, (max_width, max_height))
    return warped

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Could not load image")
        return

    mask = get_board_mask(img)
    contour, corners = find_board_corners(img, mask)

    if corners is None:
        print("Board not found")
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        return

    outlined = img.copy()
    # Draw raw contour for reference
    if contour is not None:
        cv2.drawContours(outlined, [contour], -1, (255, 0, 0), 2)
    # Draw straight rectangle from minAreaRect
    cv2.polylines(outlined, [corners.astype(int)], True, (0, 255, 0), 3)

    warped = warp_board(img, corners, border=8)

    cv2.imshow("Mask (debug)", mask)
    cv2.imshow("Detected (blue=contour, green=rect)", outlined)
    cv2.imshow("Warped Board", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
