import cv2
import os
import time

from preprocessing.corner_selection import select_board_corners
from preprocessing.preprocessing import noise_removal
from segmentation.board_isolation import warp_board
from segmentation.segmenter import Segmenter


def run(video_path: str = "preprocessing/Test3.mp4"):
    """
    Full pipeline:
    1. Load video.
    2. User selects board corners.
    3. Every frame is warped.
    4. Apply existing professor/movement filtering.
    5. Run segmentation (2 FPS).
    6. Display visualization live.
    """
    cap = cv2.VideoCapture(video_path)
    ok, first_frame = cap.read()
    if not ok:
        raise FileNotFoundError(f"Could not read video: {video_path}")

    # ---- STEP 1: USER SELECTS CORNERS ----
    print("[INFO] Select board corners...")
    corners = select_board_corners(first_frame)
    if corners is None:
        print("[INFO] Corner selection aborted.")
        return

    # ---- STEP 2: INIT SEGMENTER ----
    seg = Segmenter(use_server_mode=False)
    os.makedirs("segmentation/output_vis", exist_ok=True)

    # ---- STEP 2.1: INIT noise_removal (your existing logic)
    first_warped = warp_board(first_frame, corners)
    noise_removal.previous_frame = first_warped.copy()

    frame_idx = 0
    last_seg_time = 0
    last_filter_time = 0
    last_filtered_frame = None
    last_vis = None
    INTERVAL = 0.5   # → 2 FPS

    # Step 3: Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rectify
        warped = warp_board(frame, corners)

        # Apply professor-filter logic and throttle professor filtering to interval
        now = time.time()
        if now - last_filter_time >= INTERVAL:
            prof_filtered = noise_removal(warped)
            last_filtered_frame = prof_filtered
            last_filter_time = now
        else:
            # reuse last filtered frame to avoid blocking
            prof_filtered = last_filtered_frame


        # Throttle segmentation interval
        now = time.time()
        if now - last_seg_time >= INTERVAL:
            vis = seg.segment_array(
                prof_filtered,
                save_json=f"segmentation/output_vis/frame0.json",
                save_vis=f"segmentation/output_vis/frame0.jpg",
            )
            last_vis = vis
            last_seg_time = now
        else:
            vis = last_vis if last_vis is not None else prof_filtered

        # ---- STEP 4: SHOW VISUALIZATION LIVE ----
        cv2.imshow("ChalkStream Segmentation", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_idx} frames.")
