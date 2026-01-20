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



from diagram_vectorization.Diagram_Vectorization_Cam import (
    skeletonize_fast_skimage,
    trace_polylines,
    prune_isolated_pixels,
)
from diagram_vectorization.Diagram_Vectorization_Cam import polylines_to_diagram_content, mask_from_array

from diagram_vectorization.content_presentation import Content, ContentType, DiagramContent
import json

def run1(video_path: str = "pipeline/Test3.mp4"):
    """
    Diagram-only test pipeline (matches run() behavior + adds a diagram canvas).

    What it does:
      - Same loop/termination/throttling style as run()
      - Uses Segmenter.segment_array() to get the normal overlay visualization
      - Forces every detected region to be treated as a DIAGRAM
      - For each region crop:
          -> builds a binary mask (mask_from_array)
          -> skeletonizes
          -> traces RAW polylines (the "Figure 3" content)
          -> stores polylines in Content(DiagramContent(...))
      - Persists JSON to: segmentation/output_vis/frame0.json
      - Shows two windows live:
          1) ChalkStream Segmentation (overlay)
          2) ChalkStream Diagram Canvas (blank + drawn polylines)

    Press ESC to exit.
    """
    import json
    import os
    import time
    from datetime import datetime

    import cv2

    from preprocessing.corner_selection import select_board_corners
    from preprocessing.preprocessing import noise_removal
    from segmentation.board_isolation import warp_board
    from segmentation.segmenter import Segmenter

    # --- diagram vectorization helpers (your existing imports) ---
    from diagram_vectorization.Diagram_Vectorization_Cam import (
        skeletonize_fast_skimage,
        trace_polylines,
        prune_isolated_pixels,
    )
    from diagram_vectorization.Diagram_Vectorization_Cam import (
        polylines_to_diagram_content,  # expected: ROI-local [[x,y],...]
        mask_from_array,               # expected: BGR/gray -> 0/1 mask
    )

    from diagram_vectorization.content_presentation import Content, ContentType, DiagramContent

    # -----------------------------
    # Video open + first frame
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    ok, first_frame = cap.read()
    if not ok:
        raise FileNotFoundError(f"Could not read video: {video_path}")

    # -----------------------------
    # Corner selection (same as run())
    # -----------------------------
    print("[INFO] Select board corners...")
    corners = select_board_corners(first_frame)
    if corners is None:
        print("[INFO] Corner selection aborted.")
        cap.release()
        return

    # -----------------------------
    # Init segmenter (same as run())
    # -----------------------------
    seg = Segmenter(use_english_model=True)
    os.makedirs("segmentation/output_vis", exist_ok=True)

    # -----------------------------
    # Init noise removal (same as run())
    # -----------------------------
    first_warped = warp_board(first_frame, corners)
    noise_removal.previous_frame = first_warped.copy()

    # -----------------------------
    # Throttling (same idea as run())
    # -----------------------------
    frame_idx = 0
    last_seg_time = 0.0
    last_filter_time = 0.0

    last_filtered_frame = None
    last_vis = None
    last_canvas = None
    last_json_payload = None

    NOISE_INTERVAL = 1 / 10    # 10 FPS max filtering
    SEGMENT_INTERVAL = 1 / 2   # 2 FPS segmentation + vectorization

    # -----------------------------
    # Helper: offset ROI-local polylines into full-frame coords
    # -----------------------------
    def offset_polylines_to_global(polylines_local, x0: int, y0: int):
        """
        polylines_local: List[List[List[float]]] where each point is [x,y] in ROI-local coords.
        returns: same structure but full-frame [x,y] coords.
        """
        out = []
        for poly in polylines_local:
            out_poly = []
            for pt in poly:
                # pt = [x,y]
                out_poly.append([float(x0 + pt[0]), float(y0 + pt[1])])
            out.append(out_poly)
        return out

    # -----------------------------
    # Main loop (same termination style as run())
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rectify board
        warped = warp_board(frame, corners)

        # -------------------------
        # Professor/noise filtering (throttled)
        # -------------------------
        now = time.time()
        if now - last_filter_time >= NOISE_INTERVAL:
            prof_filtered = noise_removal(warped)
            last_filtered_frame = prof_filtered
            last_filter_time = now
        else:
            # Reuse the last filtered frame to avoid blocking
            prof_filtered = last_filtered_frame if last_filtered_frame is not None else warped

        # -------------------------
        # Segmentation + vectorization (throttled to 2 FPS)
        # -------------------------
        now = time.time()
        if now - last_seg_time >= SEGMENT_INTERVAL:
            # 1) Get overlay visualization (like run()).
            #    segment_array() ALSO updates seg.prev_regions with the scored region dicts.
            vis = seg.segment_array(
                prof_filtered,
                save_json=None,   # IMPORTANT: we will write OUR canonical JSON below
                save_vis=None,
            )
            last_vis = vis

            # 2) Pull scored regions (dicts) from segmenter state.
            #    This is the same data that would be saved by Segmenter._save_results().
            regions = seg.prev_regions  # list of dicts with "bbox", "label", etc. :contentReference[oaicite:3]{index=3}

            # 3) Create a blank white canvas (same size as warped)
            canvas = (255 * (0 * warped + 1)).astype(warped.dtype)  # fast "all white"
            # (equivalently: np.full_like(warped, 255))

            # 4) Build Content records (forced DIAGRAM)
            contents = []
            content_id = 0

            for r in regions:
                bbox = r.get("bbox", None)
                if bbox is None:
                    continue

                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)

                # Safety: skip empty crops
                if w <= 1 or h <= 1:
                    continue
                if y < 0 or x < 0 or (y + h) > warped.shape[0] or (x + w) > warped.shape[1]:
                    continue

                crop = warped[y:y + h, x:x + w]

                # -----------------------------
                # Diagram vectorization (RAW polylines)
                # -----------------------------
                # mask_from_array should produce a 0/1-ish mask from a crop (BGR or gray)
                mask = mask_from_array(crop)

                skel = skeletonize_fast_skimage(mask)
                skel = prune_isolated_pixels(skel)

                # polylines returned in (y,x) integer pixels, ROI-local
                polylines_yx, _nodes = trace_polylines(skel)

                # Convert to ROI-local [[x,y], ...] float lists (whatever your helper does)
                polylines_local_xy = polylines_to_diagram_content(polylines_yx)

                # Offset into full-frame coordinates so the orchestrator can draw on a full canvas
                polylines_global_xy = offset_polylines_to_global(polylines_local_xy, x0=x, y0=y)

                # Draw on blank canvas (polyline expects int32 Nx1x2 in (x,y) order)
                for poly in polylines_global_xy:
                    if len(poly) < 2:
                        continue
                    pts = cv2.UMat  # keep name visible in IDE for clarity (no-op usage)

                    pts_np = cv2.UMat  # no-op; just making it obvious we want numpy below
                    pts_np = None
                    import numpy as np
                    pts_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [pts_np], isClosed=False, color=(0, 0, 0), thickness=2)

                # Build canonical Content record
                # Your Content expects timestamp to be a string :contentReference[oaicite:4]{index=4}
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                diagram = DiagramContent(polylines=polylines_global_xy)

                c = Content(
                    id=content_id,
                    bbox=(x, y, w, h),
                    content_type=ContentType.DIAGRAM,   # forced for testing
                    confidence=float(r.get("confidence", 0.0)),
                    text_preview=str(r.get("text_preview", ""))[:128],
                    status=str(r.get("status", "existing")),
                    timestamp=timestamp_str,
                    diagram_content=diagram,
                )
                contents.append(c)
                content_id += 1

            # 5) Persist JSON to the SAME place run() writes visualization artifacts
            out_json_path = "segmentation/output_vis/frame0.json"
            payload = [c.to_dict() for c in contents]
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            last_json_payload = payload
            last_canvas = canvas
            last_seg_time = now

        else:
            # Reuse last results to keep UI responsive
            vis = last_vis if last_vis is not None else prof_filtered
            canvas = last_canvas if last_canvas is not None else prof_filtered

        # -------------------------
        # Live visualization (two windows)
        # -------------------------
        cv2.imshow("ChalkStream Segmentation (run1 overlay)", vis)
        cv2.imshow("ChalkStream Diagram Canvas (run1)", canvas)

        # ESC quits (same as run())
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] run1 done. Processed {frame_idx} frames. Wrote segmentation/output_vis/frame0.json")
