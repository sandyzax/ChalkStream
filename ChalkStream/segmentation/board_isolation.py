import cv2
import numpy as np


def auto_detect_board_corners(bgr):
    """
    Robust board detection that filters out full-frame contours,
    walls, lights, and irrelevant shapes.
    Returns TL, TR, BR, BL or None.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Canny edges
    edges = cv2.Canny(blur, 50, 150)

    # Close gaps in edges
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

        # Reject too small or too large
        if area < frame_area * 0.05:  # too small
            continue
        if area > frame_area * 0.80:  # too large = whole image
            continue

        # Approximate polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # We want rectangles only
        if len(approx) == 4:
            candidates.append(approx.reshape(4, 2))

    if not candidates:
        return None

    # Choose candidate with largest area
    best = max(candidates, key=lambda c: cv2.contourArea(c))

    return order_corners(best.astype("float32"))



def order_corners(pts):
    """
    Orders the four 2D points as:
      TL, TR, BR, BL
    """
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    return ordered


def warp_board(bgr, corners):
    tl, tr, br, bl = corners

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(bgr, M, (maxWidth, maxHeight))

    return warped
