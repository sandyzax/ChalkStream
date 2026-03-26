"""Board corner ordering and perspective warping utilities."""

import cv2
import numpy as np

BOARD_ASPECT_RATIO_OPTIONS = (
    (16.0, 9.0),
    (3.0, 2.0),
)


def auto_detect_board_corners(bgr):
    """Try to find the whiteboard outline from a raw camera frame."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Start with an edge image, then close small gaps so the board outline is
    # more likely to show up as one contour.
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    frame_area = w * h

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Small contours are almost never the board.
        if area < frame_area * 0.2:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # The board should still look roughly rectangular after approximation.
        if len(approx) == 4:
            candidates.append(approx.reshape(4, 2))

    if not candidates:
        return None

    # If several quadrilaterals survive, prefer the largest one.
    best = max(candidates, key=lambda c: cv2.contourArea(c))

    return order_corners(best.astype("float32"))



def order_corners(pts):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    return ordered


def estimate_board_size(corners):
    """Estimate a stable landscape board size from the selected corner geometry."""
    tl, tr, br, bl = corners

    raw_width_a = float(np.linalg.norm(br - bl))
    raw_width_b = float(np.linalg.norm(tr - tl))
    raw_height_a = float(np.linalg.norm(tr - br))
    raw_height_b = float(np.linalg.norm(tl - bl))

    raw_width = max(raw_width_a, raw_width_b, 1.0)
    raw_height = max(raw_height_a, raw_height_b, 1.0)

    # Perspective can make a landscape board look oddly tall. Use the measured
    # board area, then snap the final shape to the nearest practical landscape ratio.
    landscape_ratio = max(raw_width, raw_height) / max(min(raw_width, raw_height), 1.0)
    target_ratio = min(
        (width / height for width, height in BOARD_ASPECT_RATIO_OPTIONS),
        key=lambda ratio: abs(ratio - landscape_ratio),
    )

    target_area = raw_width * raw_height
    target_width = max(1, int(round(np.sqrt(target_area * target_ratio))))
    target_height = max(1, int(round(target_width / target_ratio)))

    return target_width, target_height


def warp_board(bgr, corners):
    """Rectify the board so downstream code can work in board coordinates."""
    maxWidth, maxHeight = estimate_board_size(corners)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(bgr, M, (maxWidth, maxHeight))

    return warped
