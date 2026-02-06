import cv2
import numpy as np

pts = np.array([
    [143, 154],  # top-left      [0]
    [580, 52],  # top-right     [1]
    [789,286],  # bottom-right  [2]
    [324,423]   # bottom-left   [3]
], dtype="float32")

widthA = np.linalg.norm(pts[3] - pts[2])
widthB = np.linalg.norm(pts[1] - pts[0])
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(pts[2] - pts[1])
heightB = np.linalg.norm(pts[3] - pts[0])
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth, 0],
    [maxWidth, maxHeight],
    [0, maxHeight]
], dtype="float32")

M = cv2.getPerspectiveTransform(pts, dst)
img = cv2.imread("g.jpg")
warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

cv2.imshow('Grayscale', warp)

cv2.imwrite("output.jpg", warp)