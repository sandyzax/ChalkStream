from datetime import datetime
import time
import cv2
import os
import json
import tarfile
import urllib.request
from paddleocr import TextRecognition, TextDetection, PaddleOCR
import numpy
from diagram_vectorization.content_presentation import Content, ContentType

# Constants defining segmentation thresholds and morphological grouping rules

MIN_AREA_PCT = 0.0005   # 0.05% of the image area (tuneable)
MAX_AREA_PCT = 0.25     # 25% of the image area (tuneable)
#MORPH_KERNEL = (15, 15)     # Morphological kernel size for region grouping (now dynamic)
OCR_INTERVAL = 5   # seconds between OCR for existing regions
CHANGE_THRESHOLD = 25  # region difference threshold
# Only OCR once region is persistent
NEW_REGION_MIN_AGE_S = 1.0      # or 2.0 if you want
NEW_REGION_MIN_HITS = 2         # "2 segmentations"

# Consecutive frames required before showing non-diagram label
STABLE_LABEL_MIN_AGE = 5

# Consecutive frames required before switching an already-displayed label (debounce/hysteresis)
FLIP_CONFIRM_FRAMES = 5

# If a stable text/equation region temporarily disappears (e.g., screen blow-up), keep it alive
KEEPALIVE_MIN_AGE = 5
MISSING_CONFIRM_FRAMES = 5

# Ignore "monster" regions that cover most of the screen (often transient camera/zoom artifacts)
MONSTER_AREA_FRAC = 0.85

# Matching and occlusion handling
MATCH_IOU_THRESH = 0.35  # IoU threshold for matching new boxes to previous regions

# Treat large/overlapping blobs as occluders during "screen blow-up" and do not treat as content
OCCLUDER_AREA_FRAC = 0.22          # if a candidate region covers >22% of frame area, consider occluder
OCCLUDER_OVERLAP_COUNT = 3         # or if it overlaps >=3 previous regions significantly
OCCLUDER_IOU_MIN = 0.12            # what counts as "significant overlap"
from paddleocr import PaddleOCR





class Segmenter:
    def __init__(self, use_english_model =True, epsilon=0.12):
        # Initialize the segmenter and load OCR model; retains state between frames
        self.use_english_model = use_english_model
        self.epsilon = epsilon # IoU threshold tolerance for region matching
        self.prev_regions = []  # Cached regions from previous frame
        self.prev_img = None   # Cached image from previous frame
        self._next_region_id = 1

        # Model configuration and paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_base = os.path.join(BASE_DIR, "ocr", "models", "rec")

        if self.use_english_model:
            self.model_name = "en_PP-OCRv5_mobile_rec_infer"
        else:
            self.model_name = "PP-OCRv5_mobile_rec_infer"

        self.model_url = (
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
            f"paddle3.0.0/{self.model_name}.tar"
        )

        self.model_path = os.path.join(self.model_base, self.model_name)

        self._ensure_model()        # Ensure the model exists locally (download if missing)
        self.recognizer = self._load_recognizer()  # Load PaddleOCR recognizer instance


    def segment_regions_only(self, img):
        """
        Return only bounding boxes for regions.
        No OCR. No cropping. No NumPy arrays.
        Output: list of (x, y, w, h)
        """
        binary = self._preprocess(img)
        regions = self._find_regions(binary, img.shape)
        return regions


    def _ensure_model(self):
        # Download and extract OCR model if not already cached
        if os.path.exists(os.path.join(self.model_path, "inference.pdiparams")):
            return
        os.makedirs(self.model_base, exist_ok=True)
        tar_path = os.path.join(self.model_base, os.path.basename(self.model_url))
        print(f"[INFO] Downloading OCR model from {self.model_url} ...")
        urllib.request.urlretrieve(self.model_url, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=self.model_base)
        os.remove(tar_path)
        print(f"[INFO] Model extracted to {self.model_base}")

    def _load_recognizer(self):
        print(f"[INFO] Loading PaddleOCR model...")

        if self.use_english_model:
            model_type = "en_PP-OCRv5_mobile_rec"
            self.model_name = "en_PP-OCRv5_mobile_rec_infer"
        else:
            model_type = "PP-OCRv5_mobile_rec"
            self.model_name = "PP-OCRv5_mobile_rec_infer"

        print(f"[INFO] Selected OCR model: {self.model_name}")

        recognizer = TextRecognition(
            model_name=model_type,
            device="gpu"
        )

        print("[INFO] OCR model loaded successfully.")
        return recognizer

    def segment_frame(self, image_path, output_json=None, output_img=None):
        # Process one image frame: detect, classify, and track regions over time
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Preprocess image to binary form and find region bounding boxes
        binary = self._preprocess(img)
        boxes = self._find_regions(binary, img.shape)

        # If this is the first frame, mark all regions as new
        if not self.prev_regions:
            print(f"[INFO] Initial frame: {len(boxes)} regions detected.")
            scored = self._score_regions(img, boxes, status="new")
        else:
            # For subsequent frames, compare each region against previous frame
            print(f"[INFO] Comparing {len(boxes)} regions to previous frame...")
            scored = self._compare_and_score(img, boxes)

        # Cache the regions for use in the next frame comparison
        self.prev_regions = scored

        # Save outputs to disk
        if output_json:
            self._save_results(scored, output_json)
        if output_img:
            vis = self._visualize(img, scored)
            cv2.imwrite(output_img, vis)

        return scored

    def segment_dict_to_content_record(seg, content_id):
        return Content(
            id=content_id,
            bbox=seg["bbox"],
            content_type=ContentType.DIAGRAM,  # forced for now
            confidence=seg.get("confidence", 0.0),
            text_preview=seg.get("text_preview", ""),
            status=seg.get("status", "new"),
            timestamp=seg["timestamp"],
        )


    def segment_array(self, img, save_json=None, save_vis=None):
        if img is None:
            raise ValueError("segment_array received None image")

        binary = self._preprocess(img)
        boxes = self._find_regions(binary, img.shape)

        if not self.prev_regions:
            scored = self._score_regions(img, boxes, status="new")
        else:
            scored = self._compare_and_score(img, boxes)

        self.prev_regions = scored
        self.prev_img = img.copy()

        # Visualization
        vis = self._visualize(img, scored)

        if save_json:
            self._save_results(scored, save_json)
        if save_vis:
            cv2.imwrite(save_vis, vis)

        return vis


    def _preprocess(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove board shadows (illumination normalization)

        # Stretch contrast (normalizes whiteboard brightness)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Light smoothing
        blur = cv2.GaussianBlur(norm, (5, 5), 0)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 12   # slightly stronger constant helps eliminate shadows
        )

        # Dynamic morphological closing
        h, w = img.shape[:2]
        k = max(3, int(min(h, w) * 0.025))  # 2.5% of board dimension
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return closed

    def _find_regions(self, binary, img_shape):
        # Detect contours representing candidate written areas
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img_shape[:2]
        total_pixels = w * h

        # Compute dynamic thresholds
        min_area = total_pixels * MIN_AREA_PCT
        max_area = total_pixels * MAX_AREA_PCT

        boxes = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh

            # Filter out noise and extreme bounding boxes
            if area < min_area:
                continue
            if area > max_area:
                continue

            boxes.append((x, y, bw, bh))

        return boxes

    def _compare_and_score(self, img, new_boxes):
        results = []
        now = time.time()

        # Reset occluder flag for this segmentation pass
        self.occluder_present = False

        # Use previous frame for change detection (if available)
        prev_img = getattr(self, "prev_img", None)

        used_prev_idxs = set()

        for (x, y, w, h) in new_boxes:
            x, y, w, h = int(x), int(y), int(w), int(h)
            new_bbox = [x, y, w, h]

            # Ignore extremely large "monster" regions (often transient zoom/blow-up artifacts)
            try:
                img_h, img_w = img.shape[:2]
                if (w * h) > (MONSTER_AREA_FRAC * (img_w * img_h)):
                    continue
            except Exception:
                pass

            # Occluder suppression:
            # During "screen blow-up" events, segmentation can produce large blobs that overlap many existing regions.
            # Treat these as occluders (instructor/zoom artifact), not real content.
            try:
                img_h, img_w = img.shape[:2]
                area_frac = (w * h) / float(img_w * img_h + 1e-6)

                overlap_count = 0
                for prev in self.prev_regions:
                    px, py, pw, ph = map(int, prev.get("bbox", (0, 0, 0, 0)))
                    # IoU with prev
                    ix1 = max(x, px); iy1 = max(y, py)
                    ix2 = min(x + w, px + pw); iy2 = min(y + h, py + ph)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    if inter <= 0:
                        continue
                    union = (w * h) + (pw * ph) - inter
                    iou_p = inter / float(union + 1e-6)
                    if iou_p >= OCCLUDER_IOU_MIN:
                        overlap_count += 1
                        if overlap_count >= OCCLUDER_OVERLAP_COUNT:
                            break

                if area_frac >= OCCLUDER_AREA_FRAC or overlap_count >= OCCLUDER_OVERLAP_COUNT:
                    self.occluder_present = True
                    # Skip creating a content region for this blob
                    continue
            except Exception:
                pass

            # -----------------------------
            # Match to previous region by best IoU
            # -----------------------------
            best_iou = 0.0
            best_prev = None
            best_prev_idx = None

            for idx, prev in enumerate(self.prev_regions):
                if idx in used_prev_idxs:
                    continue

                px, py, pw, ph = prev["bbox"]
                px, py, pw, ph = int(px), int(py), int(pw), int(ph)

                inter_x1 = max(x, px)
                inter_y1 = max(y, py)
                inter_x2 = min(x + w, px + pw)
                inter_y2 = min(y + h, py + ph)

                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                union_area = (w * h) + (pw * ph) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_prev = prev
                    best_prev_idx = idx

            matched_region = None
            if best_prev is not None and best_iou >= MATCH_IOU_THRESH:
                matched_region = best_prev
                used_prev_idxs.add(best_prev_idx)

            # -----------------------------
            # Case 1: New region => DO NOT OCR YET
            # -----------------------------
            if matched_region is None:
                results.append({
                    "bbox": new_bbox,
                    "region_id": self._next_region_id,
                    "raw_label": "diagram",      # classifier label (ungated)
                    "label": "diagram",          # gated label shown downstream
                    "age": 1,                 # consecutive frames with same raw_label
                    "display_label": "diagram",
                    "flip_streak": 0,
                    "missing_streak": 0,
                    "confidence": 0.0,
                    "text_preview": "",
                    "status": "new_pending",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "ocr_time_s": None,
                    "last_ocr_time": 0.0,        # 0 means never
                    "first_seen_time": now,
                    "seen_count": 1,
                    "has_ocr": False,
                })
                self._next_region_id += 1
                continue

            # -----------------------------
            # Case 2: Existing region => update persistence + decide whether to OCR
            # -----------------------------
            result = matched_region.copy()
            result["bbox"] = new_bbox
            result["status"] = "existing"
            result["missing_streak"] = 0
            result.setdefault("first_seen_time", now)    # in case old state lacks it
            result["seen_count"] = int(result.get("seen_count", 1)) + 1
            result.setdefault("has_ocr", bool(result.get("last_ocr_time", 0.0)))
            result.setdefault("raw_label", str(result.get("raw_label", result.get("label", "diagram"))))
            result.setdefault("age", 1)
            result.setdefault("display_label", str(result.get("display_label", result.get("label", "diagram"))))
            result.setdefault("flip_streak", 0)
            result.setdefault("missing_streak", 0)

            age = now - float(result.get("first_seen_time", now))
            has_ocr = bool(result.get("has_ocr", False))

            # Gate the FIRST OCR until region is persistent
            if not has_ocr:
                if age >= NEW_REGION_MIN_AGE_S or result["seen_count"] >= NEW_REGION_MIN_HITS:
                    ocr = self._ocr_single_region(img, new_bbox, status="new")
                    ocr["first_seen_time"] = float(result["first_seen_time"])
                    ocr["seen_count"] = int(result["seen_count"])
                    ocr["has_ocr"] = True
                    ocr["last_ocr_time"] = now
                    ocr = self._apply_stability(matched_region, ocr)
                    results.append(ocr)
                else:
                    # Still pending; keep it but don't OCR
                    result["status"] = "new_pending"
                    result = self._advance_stability_no_ocr(matched_region, result)
                    results.append(result)
                continue

            # If it already has OCR, keep your existing "changed or elapsed" behavior
            changed = False
            if prev_img is not None:
                try:
                    px, py, pw, ph = map(int, matched_region["bbox"])
                    old_crop = prev_img[py:py + ph, px:px + pw]
                    new_crop = img[y:y + h, x:x + w]
                    if old_crop.size > 0 and new_crop.size > 0:
                        old_resized = cv2.resize(old_crop, (w, h))
                        diff = cv2.absdiff(old_resized, new_crop)
                        if diff.mean() > CHANGE_THRESHOLD:
                            changed = True
                except Exception:
                    changed = True

            elapsed = now - float(matched_region.get("last_ocr_time", 0.0))
            if changed or elapsed >= OCR_INTERVAL:
                ocr = self._ocr_single_region(img, new_bbox, status="existing")
                ocr["first_seen_time"] = float(result["first_seen_time"])
                ocr["seen_count"] = int(result["seen_count"])
                ocr["has_ocr"] = True
                ocr["last_ocr_time"] = now
                ocr = self._apply_stability(matched_region, ocr)
                results.append(ocr)
            else:
                result["last_ocr_time"] = float(matched_region.get("last_ocr_time", now))
                result = self._advance_stability_no_ocr(matched_region, result)
                results.append(result)

        # ------------------------------------------------------------------
        # Keepalive for stable text/equation regions that temporarily disappear
        # (e.g., transient zoom/blow-up). We only drop them after they have
        # been missing for MISSING_CONFIRM_FRAMES consecutive segmentations.
        # ------------------------------------------------------------------
        try:
            prev_list = getattr(self, "prev_regions", []) or []
            for idx, prev in enumerate(prev_list):
                if idx in used_prev_idxs:
                    continue

                disp = str(prev.get("display_label", prev.get("label", "diagram")))
                is_keepalive = (
                    disp in ("text", "equation")
                    and int(prev.get("age", 1)) >= KEEPALIVE_MIN_AGE
                )

                if not is_keepalive:
                    continue

                ghost = prev.copy()
                ghost["missing_streak"] = int(prev.get("missing_streak", 0)) + 1
                ghost["status"] = "missing"

                if ghost["missing_streak"] < MISSING_CONFIRM_FRAMES:
                    # Hold last known display label/content
                    ghost["label"] = disp
                    ghost["display_label"] = disp
                    ghost["flip_streak"] = 0
                    results.append(ghost)
        except Exception:
            pass


        return results




    
    def _update_region_state(self, prev_region: dict, cur_region: dict, raw_known: bool) -> dict:
        """Update persistence fields (raw_label age) and debounced display label.

        raw_known=True  -> cur_region contains a newly computed raw_label for this frame (from OCR/classifier)
        raw_known=False -> carry forward prev_region raw_label (no OCR this frame)

        Fields:
          - raw_label: instantaneous classifier output (or carried forward)
          - age: consecutive frames where raw_label stayed the same
          - display_label: what we currently show downstream
          - flip_streak: consecutive frames candidate_label disagrees with display_label
          - label: alias of display_label for downstream compatibility
        """
        prev_raw = str(prev_region.get("raw_label", prev_region.get("label", "diagram")))
        prev_age = int(prev_region.get("age", 1))

        if raw_known:
            cur_raw = str(cur_region.get("raw_label", cur_region.get("label", "diagram")))
        else:
            cur_raw = prev_raw

        # Update age based on raw_label stability
        if cur_raw == prev_raw:
            age = prev_age + 1
        else:
            age = 1

        cur_region["raw_label"] = cur_raw
        cur_region["age"] = age

        # Candidate label (promotion delay): force diagram until region is old enough
        candidate_label = "diagram" if age < STABLE_LABEL_MIN_AGE else cur_raw

               # Debounce label flips (including text->diagram and diagram->text)
        prev_display = str(prev_region.get("display_label", prev_region.get("label", "diagram")))
        prev_flip = int(prev_region.get("flip_streak", 0))

        # -----------------------------
        # NEW: equation stickiness
        # If we're currently displaying "equation", require more consecutive
        # evidence to demote away from equation.
        # -----------------------------
        promote_confirm = FLIP_CONFIRM_FRAMES
        demote_confirm = FLIP_CONFIRM_FRAMES

        if prev_display == "equation" and candidate_label != "equation":
            demote_confirm = FLIP_CONFIRM_FRAMES * 2   # e.g., 10 frames instead of 5

        # Optional micro-guard: ignore low-confidence demotions when raw is known
        # (keeps equations from flickering due to garbage OCR reads)
        if raw_known and prev_display == "equation" and candidate_label != "equation":
            cur_conf = float(cur_region.get("confidence", 0.0))
            if cur_conf < 0.70:
                candidate_label = "equation"

        if candidate_label == prev_display:
            display_label = prev_display
            flip_streak = 0
        else:
            flip_streak = prev_flip + 1
            needed = demote_confirm if (prev_display == "equation" and candidate_label != "equation") else promote_confirm
            if flip_streak >= needed:
                display_label = candidate_label
                flip_streak = 0
            else:
                display_label = prev_display


        
        # Persist stability state
        cur_region["display_label"] = display_label
        cur_region["flip_streak"] = flip_streak
        # What downstream sees
        cur_region["label"] = display_label
        return cur_region

    def _apply_stability(self, prev_region: dict, cur_region: dict) -> dict:
        # OCR/classifier ran this frame; raw_label is known
        return self._update_region_state(prev_region, cur_region, raw_known=True)

    def _advance_stability_no_ocr(self, prev_region: dict, cur_region: dict) -> dict:
        # No OCR/classifier this frame; carry forward raw_label
        return self._update_region_state(prev_region, cur_region, raw_known=False)


    def _ocr_single_region(self, img, box, status):
        # Run OCR inference on one cropped region and log timing details
        x, y, w, h = box
        crop = img[y:y+h, x:x+w]
        tmp_path = "_tmp_crop.png"
        cv2.imwrite(tmp_path, crop)

        start_time = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        try:
            output = self.recognizer.predict(tmp_path)
            elapsed = time.perf_counter() - start_time

            # Parse recognition result
            result = output[0] if isinstance(output[0], dict) else vars(output[0])
            text = result.get("rec_text", "")
            conf = result.get("rec_score", 0.0)

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            text, conf = "", 0.0
            print(
                f"[{timestamp}] [ERROR] OCR failed ({x},{y},{w},{h}) "
                f"after {elapsed:.3f}s: {e}"
            )
            text, conf = "", 0.0

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        label = self._classify(text, conf, crop=crop, box=box)

        print(
            f"[{timestamp}] [INFO] OCR region ({x},{y},{w},{h}) "
            f"took {elapsed:.3f}s | label={label} | conf={conf:.3f} | text=\"{text[:64]}\""
        )
        
        return {
            "bbox": [x, y, w, h],
            "raw_label": label,
            "label": label,
            "confidence": round(conf, 3),
            "text_preview": text,
            "status": status,
            "timestamp": timestamp,
            "ocr_time_s": round(elapsed, 3)
        }

    def _score_regions(self, img, boxes, status="new"):
        # Run OCR on all regions with a uniform status (used for first frame)
        return [self._ocr_single_region(img, b, status) for b in boxes]

    def _classify(self, text, conf, crop = None, box=None):
        # Lightweight content classifier based on recognized text and confidence\
        if box is not None:
            _, _, w, h = map(int, box)
            if w > 0 and h > 0:
                aspect = max(w / float(h), h / float(w))
                if aspect >= 10 and min(w, h) >= 6:
                    return "diagram"
        if crop is not None : # probable equation
            if any(sym in text for sym in ["=", "+", "-", "∫", "√", "π", "Σ", "\\", "$", "÷", "??"]):
                return "equation"
            if self._detect_fraction_bar(crop):
                return "equation"
        if (conf > 0.65) & (len(text) >= 2): # lower bound probable text, likely not equation but might be diagram 
            return "text"
        
        return "diagram"

    def _visualize(self, img, regions):
        # Draw bounding boxes and class labels on the image (unchanged)
        vis = img.copy()
        for r in regions:
            x, y, w, h = r["bbox"]
            label = r["label"]
            conf = r["confidence"]
            color = (
                (0, 255, 0) if label == "text" else
                (255, 0, 0) if label == "equation" else
                (255, 0, 255)
            )
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis, f"{label} ({conf:.2f})",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        return vis

    def _save_results(self, data, path):
        # Write detected regions and metadata to JSON
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved {len(data)} regions to {path}")


    def _detect_fraction_bar(self, crop):
        """
        DEBUG VERSION
        Detects a long, near-continuous horizontal fraction bar.
        Prints detailed diagnostics for every rejection.
        """

        import cv2
        import numpy as np

        print("\n================ FRACTION BAR DEBUG ================")
        print(f"[FB] Crop shape: {crop.shape}")

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(blur, 60, 160)

        h, w = edges.shape
        print(f"[FB] Edge image size: w={w}, h={h}")

        # ---- Tunables ----
        MIN_LEN_FRAC = 0.60
        MAX_GAP_PX = 10
        MAX_SLOPE = 0.15
        CENTER_BAND = (0.20, 0.80)

        print(f"[FB] Params:")
        print(f"     MIN_LEN_FRAC={MIN_LEN_FRAC}")
        print(f"     MAX_GAP_PX={MAX_GAP_PX}")
        print(f"     MAX_SLOPE={MAX_SLOPE}")
        print(f"     CENTER_BAND={CENTER_BAND}")

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            minLineLength=int(w * MIN_LEN_FRAC),
            threshold=int((w * MIN_LEN_FRAC) * 0.3 - 0.5),
            maxLineGap=MAX_GAP_PX
        )

        if lines is None:
            print("[FB] ❌ No Hough lines detected")
            return False

        print(f"[FB] Hough returned {len(lines)} candidate lines")

        def continuity_score(edges_img, y, x1, x2, band=2):
            x1, x2 = sorted((int(x1), int(x2)))
            y = int(round(y))
            y0 = max(0, y - band)
            y1 = min(edges_img.shape[0], y + band + 1)
            strip = edges_img[y0:y1, x1:x2]

            if strip.size == 0:
                return 0.0

            col_has_edge = (strip > 0).any(axis=0)
            score = col_has_edge.mean()
            return score

        for idx, (x1, y1, x2, y2) in enumerate(lines[:, 0]):
            print(f"\n[FB] ---- Candidate {idx} ----")
            print(f"[FB] Raw line: ({x1},{y1}) -> ({x2},{y2})")

            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            slope = abs(dy / (dx + 1e-6))
            mid_y = (y1 + y2) / 2.0

            print(f"[FB] Length: {length:.1f}px (min {w * MIN_LEN_FRAC:.1f}px)")
            if length < w * MIN_LEN_FRAC:
                print("[FB] ❌ Rejected: too short")
                continue

            print(f"[FB] Slope: {slope:.3f} (max {MAX_SLOPE})")
            if slope > MAX_SLOPE:
                print("[FB] ❌ Rejected: too slanted")
                continue

            print(f"[FB] Mid Y: {mid_y:.1f}px (allowed {h * CENTER_BAND[0]:.1f}–{h * CENTER_BAND[1]:.1f})")
            if not (h * CENTER_BAND[0] <= mid_y <= h * CENTER_BAND[1]):
                print("[FB] ❌ Rejected: outside center band")
                continue

            cont = continuity_score(edges, mid_y, x1, x2, band=2)
            print(f"[FB] Continuity score: {cont:.3f} (min 0.85)")

            if cont < 0.85:
                print("[FB] ❌ Rejected: insufficient continuity")
                continue

            print("[FB] ✅ FRACTION BAR DETECTED")
            return True

        print("[FB] ❌ No candidates passed all checks")
        return False
