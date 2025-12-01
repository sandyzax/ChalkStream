from datetime import datetime
import time
import cv2
import os
import json
import tarfile
import urllib.request
from paddleocr import TextRecognition

# Constants defining segmentation thresholds and morphological grouping rules

MIN_AREA_PCT = 0.0005   # 0.05% of the image area (tuneable)
MAX_AREA_PCT = 0.25     # 25% of the image area (tuneable)
#MORPH_KERNEL = (15, 15)     # Morphological kernel size for region grouping (now dynamic)
OCR_INTERVAL = 5   # seconds between OCR for existing regions
CHANGE_THRESHOLD = 25  # region difference threshold



class Segmenter:
    def __init__(self, use_english_model =True, epsilon=0.05):
        # Initialize the segmenter and load OCR model; retains state between frames
        self.use_english_model = use_english_model
        self.epsilon = epsilon              # IoU threshold tolerance for region matching
        self.prev_regions = []              # Cached regions from previous frame

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
        for (x, y, w, h) in new_boxes: 
            new_bbox = [x, y, w, h] 
            matched_region = None # Try to match with a previous region using IoU 
            for prev in self.prev_regions: 
                px, py, pw, ph = prev["bbox"] 
                
                # IoU 
                inter_x1 = max(x, px) 
                inter_y1 = max(y, py) 
                inter_x2 = min(x + w, px + pw) 
                inter_y2 = min(y + h, py + ph) 
                inter_w = max(0, inter_x2 - inter_x1) 
                inter_h = max(0, inter_y2 - inter_y1) 
                inter_area = inter_w * inter_h 
                union_area = (w * h) + (pw * ph) - inter_area 
                iou = inter_area / union_area if union_area > 0 else 0 
                
                if iou > (1 - self.epsilon): 
                    matched_region = prev 
                    break 
                
            # Case 1: New Region, Always OCR 
            if matched_region is None: 
                result = self._ocr_single_region(img, new_bbox, status="new") 
                result["last_ocr_time"] = now 
                results.append(result) 
                continue 
            
            # Case 2: Existing Region, may skip OCR 
            # # check visual change 
            changed = False 
            try: 
                old_crop = img[ matched_region["bbox"][1] : matched_region["bbox"][1] + matched_region["bbox"][3],
                    matched_region["bbox"][0] : matched_region["bbox"][0] + matched_region["bbox"][2] 
                ] 
                new_crop = img[y:y+h, x:x+w] 
                diff = cv2.absdiff(cv2.resize(old_crop, (w, h)), new_crop) 
                if diff.mean() > CHANGE_THRESHOLD: 
                    changed = True 
            except: 
                pass 
            
            elapsed = now - matched_region.get("last_ocr_time", 0) 
            
            # Decide whether to OCR again 
            if changed or elapsed >= OCR_INTERVAL: 
                # Re-run OCR 
                result = self._ocr_single_region(img, new_bbox, status="existing") 
                result["last_ocr_time"] = now 
            else: 
                # Skip OCR, reuse previous detection 
                result = matched_region.copy() 
                result["bbox"] = new_bbox 
                result["status"] = "existing" 
            
            results.append(result) 
            
        return results


    def _compare_and_score(self, img, new_boxes):
        results = []
        now = time.time()

        for (x, y, w, h) in new_boxes:
            new_bbox = [x, y, w, h]
            matched_region = None

            # Try to match with a previous region using IoU
            for prev in self.prev_regions:
                px, py, pw, ph = prev["bbox"]

                # IoU
                inter_x1 = max(x, px)
                inter_y1 = max(y, py)
                inter_x2 = min(x + w, px + pw)
                inter_y2 = min(y + h, py + ph)
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                union_area = (w * h) + (pw * ph) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                if iou > (1 - self.epsilon):
                    matched_region = prev
                    break
                
            # Case 1: New Region, Always OCR
            if matched_region is None:
                result = self._ocr_single_region(img, new_bbox, status="new")
                result["last_ocr_time"] = now
                results.append(result)
                continue

            # Case 2: Existing Region, may skip OCR

            # check visual change
            changed = False
            try:
                old_crop = img[
                    matched_region["bbox"][1] : matched_region["bbox"][1] + matched_region["bbox"][3],
                    matched_region["bbox"][0] : matched_region["bbox"][0] + matched_region["bbox"][2]
                ]
                new_crop = img[y:y+h, x:x+w]
                diff = cv2.absdiff(cv2.resize(old_crop, (w, h)), new_crop)
                if diff.mean() > CHANGE_THRESHOLD:
                    changed = True
            except:
                pass

            elapsed = now - matched_region.get("last_ocr_time", 0)

            # Decide whether to OCR again
            if changed or elapsed >= OCR_INTERVAL:
                # Re-run OCR
                result = self._ocr_single_region(img, new_bbox, status="existing")
                result["last_ocr_time"] = now
            else:
                # Skip OCR → reuse previous detection
                result = matched_region.copy()
                result["bbox"] = new_bbox
                result["status"] = "existing"

            results.append(result)

        return results

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

            # Determine label and console message based on confidence
            label_message = "Text: " + text[:128] if conf > 0.625 else "Diagram"

            print(
                f"[{timestamp}] [INFO] OCR region ({x},{y},{w},{h}) "
                f"took {elapsed:.3f}s | conf={conf:.3f} | {label_message}"
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{timestamp}] [ERROR] OCR failed ({x},{y},{w},{h}) "
                f"after {elapsed:.3f}s: {e}"
            )
            text, conf = "", 0.0

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        label = self._classify(text, conf)
        return {
            "bbox": [x, y, w, h],
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

    def _classify(self, text, conf):
        # Lightweight content classifier based on recognized text and confidence
        if any(sym in text for sym in ["=", "+", "-", "∫", "√", "π", "Σ"]):
            return "equation"
        if conf > 0.65:
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

