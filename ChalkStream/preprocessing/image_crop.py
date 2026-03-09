import cv2
import numpy as np
from pathlib import Path

pts = np.array([
    [101, 86],  # top-left      [0]
    [538, 87],  # top-right     [1]
    [539,297],  # bottom-right  [2]
    [103,303]   # bottom-left   [3]
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
REPO_ROOT = Path(__file__).resolve().parent.parent
img = cv2.imread(str(REPO_ROOT / "assets" / "samples" / "input.jpg"))
warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

cv2.imshow('Grayscale', warp)
