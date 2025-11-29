"""
Utilities for selecting and confirming the four board corners.
Auto-detect if possible, then let the user drag handles to adjust.
"""
import cv2
import numpy as np
from typing import Optional

from segmentation.board_isolation import auto_detect_board_corners


class CornerEditor:
    """Simple OpenCV UI to drag 4 points and confirm/reset/abort."""

    def __init__(self, frame: np.ndarray, initial_corners: np.ndarray):
        self.frame = frame
        self.corners = initial_corners.astype(np.float32)
        self.dragging_idx: Optional[int] = None
        self.window = "Adjust board corners (Enter=confirm, r=reset, q=abort)"
        self.radius = 12

    def _draw_overlay(self) -> np.ndarray:
        vis = self.frame.copy()
        cv2.polylines(vis, [self.corners.astype(int)], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(self.corners):
            cv2.circle(vis, (int(x), int(y)), self.radius, (0, 165, 255), -1)
            cv2.putText(
                vis, str(i), (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )
        cv2.putText(
            vis,
            "Drag handles -> Enter confirm | r reset | q abort",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return vis

    def _nearest_corner(self, x: int, y: int) -> Optional[int]:
        dists = np.linalg.norm(self.corners - np.array([x, y], dtype=np.float32), axis=1)
        idx = int(np.argmin(dists))
        return idx if dists[idx] <= self.radius * 1.5 else None

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._nearest_corner(x, y)
            if idx is not None:
                self.dragging_idx = idx
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_idx is not None:
            self.corners[self.dragging_idx] = (float(x), float(y))
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_LBUTTONDBLCLK):
            self.dragging_idx = None

    def edit(self) -> Optional[np.ndarray]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1600, 900)
        cv2.setMouseCallback(self.window, self._on_mouse)
        original = self.corners.copy()
        while True:
            vis = self._draw_overlay()
            cv2.imshow(self.window, vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):  # Enter
                break
            elif key == ord("r"):
                self.corners = original.copy()
            elif key == ord("q") or key == 27:
                self.corners = None
                break
        cv2.destroyWindow(self.window)
        from segmentation.board_isolation import order_corners
        return order_corners(self.corners.astype(np.float32))


def _initial_corners_from_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.array(
        [
            [0.05 * w, 0.05 * h],
            [0.95 * w, 0.05 * h],
            [0.95 * w, 0.95 * h],
            [0.05 * w, 0.95 * h],
        ],
        dtype=np.float32,
    )


def auto_detect_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    return auto_detect_board_corners(frame)



def select_board_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Try auto-detect; open drag UI to confirm/adjust.
    Returns 4x2 float32 corners (TL, TR, BR, BL) or None if aborted.
    """
    # 1. Auto-detect corners (if fails, fallback to box)
    try:
        auto_corners = auto_detect_corners(frame)
    except Exception as exc:
        print(f"[WARN] Auto-detection failed: {exc}")
        auto_corners = None

    # 2. Initial guess
    initial = auto_corners if auto_corners is not None else _initial_corners_from_frame(frame)

    # 3. Go directly into the draggable editor
    editor = CornerEditor(frame, initial)
    corners = editor.edit()

    if corners is None:
        print("[INFO] Corner selection aborted by user.")
        return None

    # 4. Always return ordered corners (TL, TR, BR, BL)
    from segmentation.board_isolation import order_corners
    return order_corners(corners.astype(np.float32))

