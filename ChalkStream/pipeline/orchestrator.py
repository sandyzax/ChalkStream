import cv2
import os
import time

from preprocessing.corner_selection import select_board_corners
from preprocessing.preprocessing import noise_removal
from segmentation.board_isolation import warp_board
from segmentation.segmenter import Segmenter


def run(video_path: str = "pipeline/Test3.mp4"):
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

    # Step 1: User selects board corners
    print("[INFO] Select board corners...")
    corners = select_board_corners(first_frame)
    if corners is None:
        print("[INFO] Corner selection aborted.")
        return

    # Step 2: Initialize segmentation
    seg = Segmenter(use_english_model= True)
    os.makedirs("segmentation/output_vis", exist_ok=True)

    # Step 2.1: Initialize noise_removal 
    first_warped = warp_board(first_frame, corners)
    noise_removal.previous_frame = first_warped.copy()

    frame_idx = 0
    last_seg_time = 0
    last_filter_time = 0
    last_filtered_frame = None
    last_vis = None
    NOISE_INTERVAL = 1/10  # 10 FPS max
    SEGMENT_INTERVAL = 1/2   # 2 FPS

    # Step 3: Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rectify
        warped = warp_board(frame, corners)

        # Apply professor-filter logic and throttle professor filtering to interval
        now = time.time()
        if now - last_filter_time >= NOISE_INTERVAL:
            prof_filtered = noise_removal(warped)
            last_filtered_frame = prof_filtered
            last_filter_time = now
        else:
            # reuse last filtered frame to avoid blocking
            prof_filtered = last_filtered_frame


        # Throttle segmentation interval
        now = time.time()
        if now - last_seg_time >= SEGMENT_INTERVAL:
            vis = seg.segment_array(
                prof_filtered,
                save_json=f"segmentation/output_vis/frame0.json",
                save_vis=f"segmentation/output_vis/frame0.jpg",
            )
            last_vis = vis
            last_seg_time = now
        else:
            vis = last_vis if last_vis is not None else prof_filtered

        # Step 4: Show visualization live
        cv2.imshow("ChalkStream Segmentation", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_idx} frames.")





def run1(video_path: any = 0):
    """
    Diagram-only test pipeline that works the same for:
      - Live capture (webcam / laptop camera): video_path is int (e.g., 0)
      - Video files (.mp4): video_path is str path

    Output:
      - Two resizable OpenCV windows
      - Writes JSON: segmentation/output_vis/frame0.json
      - Prints periodic stats to terminal

    MOD: when a FILE video ends, hold on the final displayed frames until ESC.
    """
    import os
    import json
    import time
    from datetime import datetime

    import cv2
    import numpy as np

    from preprocessing.mp4_audio_playback import start_audio_playback
    from preprocessing.corner_selection import select_board_corners
    from preprocessing.preprocessing import noise_removal
    from segmentation.board_isolation import warp_board
    from segmentation.segmenter import Segmenter

    from diagram_vectorization.Diagram_Vectorization_Cam import (
        skeletonize_fast_skimage,
        trace_polylines,
        prune_isolated_pixels,
        polylines_to_diagram_content,
        mask_from_array,
    )
    from diagram_vectorization.content_presentation import (
        Content, ContentType, DiagramContent, draw_text_in_bbox
    )

    # -----------------------------
    # Helper: offset ROI-local polylines into full-frame coords
    # -----------------------------
    def offset_polylines_to_global(polylines_local, x0: int, y0: int):
        out = []
        for poly in polylines_local:
            out_poly = []
            for pt in poly:
                out_poly.append([float(x0 + pt[0]), float(y0 + pt[1])])
            out.append(out_poly)
        return out

    # -----------------------------
    # Input type + capture
    # -----------------------------
    is_live = isinstance(video_path, int)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    # FPS / timing for playback (file only; live uses camera pacing)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0  # fallback (common for webcams / some files)
    frame_period = 1.0 / fps

    # Playback clock (used only for file pacing)
    playback_t0 = time.time()
    frame_no = 0  # frames advanced in the stream timeline (includes skipped frames)

    # -----------------------------
    # Get first frame for corner selection
    # -----------------------------
    def get_first_frame(cap, is_live, video_path):
        if not is_live:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read first frame from file: {video_path}")
            return frame

        print("[INFO] Camera preview: orient the laptop/camera.")
        print("       Press SPACE/ENTER to freeze a frame for corner selection.")
        print("       Press ESC or Q to quit.")

        cv2.namedWindow("Camera Preview (press SPACE to select corners)", cv2.WINDOW_NORMAL)

        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read from live camera.")

            cv2.imshow("Camera Preview (press SPACE to select corners)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (32, 13):  # SPACE or ENTER
                cv2.destroyWindow("Camera Preview (press SPACE to select corners)")
                return frame.copy()

            if key in (27, ord("q"), ord("Q")):  # ESC or q
                cap.release()
                cv2.destroyAllWindows()
                print("[INFO] Aborted during preview.")
                return None

    first_frame = get_first_frame(cap, is_live, video_path)
    if first_frame is None:
        return

    # -----------------------------
    # Corner selection
    # -----------------------------
    print("[INFO] Select board corners...")
    corners = select_board_corners(first_frame)
    if corners is None:
        print("[INFO] Corner selection aborted.")
        cap.release()
        return

    # -----------------------------
    # Init segmenter + output dir
    # -----------------------------
    seg = Segmenter(use_english_model=True)
    os.makedirs("segmentation/output_vis", exist_ok=True)

    # -----------------------------
    # Init noise removal baseline
    # -----------------------------
    first_warped = warp_board(first_frame, corners)
    noise_removal.previous_frame = first_warped.copy()

    # -----------------------------
    # Throttling
    # -----------------------------
    processed_frames = 0
    last_seg_time = 0.0
    last_filter_time = 0.0
    last_filtered_frame = None
    last_vis = None
    last_canvas = None

    NOISE_INTERVAL = 1 / 10     # 10 FPS max filtering
    SEGMENT_INTERVAL = 1 / 2    # 2 FPS segmentation + vectorization

    # -----------------------------
    # Frame fetch helper (isolates live vs file pacing)
    # -----------------------------
    def next_frame(cap, is_live, frame_period, playback_t0, frame_no):
        if is_live:
            ret, frame = cap.read()
            return ret, frame, frame_no + 1

        # ---- FILE playback pacing ----
        now = time.time()
        target_time = playback_t0 + frame_no * frame_period
        lag = now - target_time

        # If behind, skip frames quickly without decode
        if lag > frame_period:
            frames_to_skip = int(lag / frame_period)
            for _ in range(frames_to_skip):
                if not cap.grab():
                    return False, None, frame_no
                frame_no += 1

        # If ahead, sleep until it's time
        now = time.time()
        target_time = playback_t0 + frame_no * frame_period
        ahead = target_time - now
        if ahead > 0:
            time.sleep(ahead)

        ret, frame = cap.read()
        return ret, frame, frame_no + 1

    # -----------------------------
    # Create resizable windows + place them
    # -----------------------------
    cv2.namedWindow("ChalkStream Segmentation (run1 overlay)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ChalkStream Diagram Canvas (run1)", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("ChalkStream Segmentation (run1 overlay)", 830, 450)
    cv2.resizeWindow("ChalkStream Diagram Canvas (run1)", 830, 450)

    cv2.moveWindow("ChalkStream Segmentation (run1 overlay)", 0, 0)
    cv2.moveWindow("ChalkStream Diagram Canvas (run1)", 0, 450)

    # -----------------------------
    # RESET playback clock *after* expensive startup (FILE only)
    # -----------------------------
    if not is_live:
        playback_t0 = time.time()
        frame_no = 0

    audio = None
    #if not is_live:
        #audio = start_audio_playback(video_path)  # async; may return None

    # Start stats clock AFTER expensive init
    start_time = time.time()
    last_print_time = 0.0
    PRINT_INTERVAL = 1.0

    # -----------------------------
    # Main loop
    # -----------------------------
    user_requested_exit = False

    while True:
        ret, frame, frame_no = next_frame(cap, is_live, frame_period, playback_t0, frame_no)
        if not ret:
            break

        # Rectify board
        warped = warp_board(frame, corners)

        # -------------------------
        # Noise filtering (throttled)
        # -------------------------
        now = time.time()
        if now - last_filter_time >= NOISE_INTERVAL:
            prof_filtered = noise_removal(warped)
            last_filtered_frame = prof_filtered
            last_filter_time = now
        else:
            prof_filtered = last_filtered_frame if last_filtered_frame is not None else warped

        # -------------------------
        # Segmentation + vectorization (throttled)
        # -------------------------
        now = time.time()
        if now - last_seg_time >= SEGMENT_INTERVAL:
            # Segmenter overlay
            vis = seg.segment_array(
                prof_filtered,
                save_json=None,
                save_vis=None,
            )
            last_vis = vis
            skip_rebuild = False
            if getattr(seg, "occluder_present", False) and last_canvas is not None:
                # Freeze the last canvas during blow-up/occlusion to avoid overlapping artifacts.
                skip_rebuild = True
                canvas = last_canvas


            if not skip_rebuild:
                regions = getattr(seg, "prev_regions", []) or []
    
                # Blank white canvas
                canvas = np.full_like(warped, 255)
    
                contents = []
                content_id = 0
    
                for r in regions:
                    label = str(r.get("label", "")).lower()
                    text_preview = str(r.get("text_preview", "")).strip()
                    bbox = r.get("bbox", None)
                    if bbox is None:
                        continue
    
                    x, y, w, h = map(int, bbox)
    
                    # Safety: skip empty/out-of-bounds crops
                    if w <= 1 or h <= 1:
                        continue
                    if x < 0 or y < 0 or (x + w) > warped.shape[1] or (y + h) > warped.shape[0]:
                        continue
    
                    if label == "text" and text_preview:
                        # 1) Render text directly
                        draw_text_in_bbox(
                            canvas,
                            text_preview,
                            (x, y, w, h),
                            height_fill=0.70,   # tune
                            margin_frac=0.08,   # tune
                            wrap=True,
                        )
    
                        # 2) Build a WORDS content record
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        c = Content(
                            id=content_id,
                            bbox=(x, y, w, h),
                            content_type=ContentType.WORDS,
                            confidence=float(r.get("confidence", 0.0)),
                            text_preview=text_preview[:128],
                            status=str(r.get("status", "existing")),
                            timestamp=timestamp_str,
                            ocr_time_s=r.get("ocr_time_s", None),
                            last_ocr_time=r.get("last_ocr_time", None),
                            age=int(r.get("age", 1)),
                        )
                        contents.append(c)
                        content_id += 1
                        continue
    
                    # Otherwise: equation/diagram => keep vectorization
                    crop = warped[y:y + h, x:x + w]
    
                    mask = mask_from_array(crop)
                    skel = skeletonize_fast_skimage(mask)
                    skel = prune_isolated_pixels(skel)
    
                    polylines_yx, _nodes = trace_polylines(skel)
                    polylines_local_xy = polylines_to_diagram_content(polylines_yx)
                    polylines_global_xy = offset_polylines_to_global(polylines_local_xy, x0=x, y0=y)
    
                    for poly in polylines_global_xy:
                        if len(poly) < 2:
                            continue
                        pts_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(canvas, [pts_np], isClosed=False, color=(0, 0, 0), thickness=2)
    
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    diagram = DiagramContent(polylines=polylines_global_xy)
    
                    ctype = ContentType.MATH if label.startswith("equation") else ContentType.DIAGRAM
    
                    c = Content(
                        id=content_id,
                        bbox=(x, y, w, h),
                        content_type=ctype,
                        confidence=float(r.get("confidence", 0.0)),
                        text_preview=text_preview[:128],
                        status=str(r.get("status", "existing")),
                        timestamp=timestamp_str,
                        diagram_content=diagram,
                        age=int(r.get("age", 1)),
                    )
                    contents.append(c)
                    content_id += 1
    
                # Persist JSON
                out_json_path = "segmentation/output_vis/frame0.json"
                payload = [c.to_dict() for c in contents]
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
    
            last_canvas = canvas
            last_seg_time = now
        else:
            # Reuse last results to keep UI responsive
            vis = last_vis if last_vis is not None else prof_filtered
            canvas = last_canvas if last_canvas is not None else prof_filtered

        # -------------------------
        # Display
        # -------------------------
        cv2.imshow("ChalkStream Segmentation (run1 overlay)", vis)
        cv2.imshow("ChalkStream Diagram Canvas (run1)", canvas)

        # ESC quits
        if cv2.waitKey(1) & 0xFF == 27:
            user_requested_exit = True
            break

        processed_frames += 1

        # periodic stats
        now = time.time()
        if now - last_print_time >= PRINT_INTERVAL:
            elapsed = now - start_time
            proc_fps = processed_frames / max(elapsed, 1e-6)

            timeline_fps = frame_no / max(elapsed, 1e-6)

            src = "LIVE" if is_live else "FILE"
            print(
                f"[STATS][{src}] t={elapsed:6.1f}s | processed={processed_frames} (proc_fps={proc_fps:5.1f}) "
                f"| timeline_advanced={frame_no} (timeline_fps={timeline_fps:5.1f})"
            )
            last_print_time = now

    # -----------------------------
    # Hold final frame on FILE end (not live), unless user already quit early
    # -----------------------------
    if (not is_live) and (not user_requested_exit):
        print("[INFO] End of video. Holding final frame. Press ESC to close.")
        while True:
            if last_vis is not None:
                cv2.imshow("ChalkStream Segmentation (run1 overlay)", last_vis)
            if last_canvas is not None:
                cv2.imshow("ChalkStream Diagram Canvas (run1)", last_canvas)

            if (cv2.waitKey(30) & 0xFF) == 27:
                break

    if audio is not None:
        audio.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] run1 done. Wrote segmentation/output_vis/frame0.json")
