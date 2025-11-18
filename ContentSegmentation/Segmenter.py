from datetime import datetime

import time
import cv2
import os
import json
import tarfile
import urllib.request
import paddle
print(paddle.version.cuda())
print(paddle.device.is_compiled_with_cuda())
print(paddle.device.get_device())

from paddleocr import TextRecognition

# Constants defining segmentation thresholds and morphological grouping rules
MIN_AREA = 800              # Minimum area in pixels for a valid region
MAX_AREA = 0.3              # Maximum relative area (fraction of total image)
MORPH_KERNEL = (12, 12)     # Morphological kernel size for region grouping


class Segmenter:
    def __init__(self, use_server_mode=False, epsilon=0.05):
        # Initialize the segmenter and load OCR model; retains state between frames
        self.use_server_mode = use_server_mode
        self.epsilon = epsilon              # IoU threshold tolerance for region matching
        self.prev_regions = []              # Cached regions from previous frame

        # Model configuration and paths
        self.model_base = os.path.join(os.getcwd(), "models", "rec")
        self.model_name = f"PP-OCRv5_{'server' if use_server_mode else 'mobile'}_rec_infer"
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
        # Create a PaddleOCR text recognizer based on model type
        print(f"[INFO] Loading PaddleOCR model ({self.model_name}) ...")
        model_type = f"PP-OCRv5_{'server' if self.use_server_mode else 'mobile'}_rec"
        recognizer = TextRecognition(model_name=model_type, device= "gpu:1")
        print(f"[INFO] OCR model loaded successfully.")
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

        # Optionally save outputs to disk
        if output_json:
            self._save_results(scored, output_json)
        if output_img:
            vis = self._visualize(img, scored)
            cv2.imwrite(output_img, vis)

        return scored

    def _preprocess(self, img):
        # Convert to grayscale, blur, and binarize to emphasize written strokes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        # Morphological closing to join disconnected strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def _find_regions(self, binary, img_shape):
        # Detect contours representing candidate written areas
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img_shape[:2]
        max_pixels = w * h
        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            # Filter out noise and extreme bounding boxes
            if area < MIN_AREA or area > MAX_AREA * max_pixels:
                continue
            if bw > w * 0.75:
                continue
            boxes.append((x, y, bw, bh))
        return boxes

    def _compare_and_score(self, img, new_boxes):
        # Compare current boxes with cached ones to determine new/existing regions
        prev_boxes = [r["bbox"] for r in self.prev_regions]
        results = []

        for (x, y, w, h) in new_boxes:
            new_area = w * h
            matched = False

            # Check overlap (Intersection of Union) between each new box and previous boxes
            for pb in prev_boxes:
                px, py, pw, ph = pb
                inter_x1 = max(x, px)
                inter_y1 = max(y, py)
                inter_x2 = min(x + w, px + pw)
                inter_y2 = min(y + h, py + ph)
                inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                union_area = new_area + (pw * ph) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                # Mark as existing if overlap exceeds 1 - epsilon
                if iou > (1 - self.epsilon):
                    matched = True
                    break

            status = "existing" if matched else "new"
            result = self._ocr_single_region(img, (x, y, w, h), status)
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
            output = self.recognizer.predict(input=tmp_path)
            elapsed = time.perf_counter() - start_time

            # Parse recognition result
            result = output[0] if isinstance(output[0], dict) else vars(output[0])
            text = result.get("rec_text", "")
            conf = result.get("rec_score", 0.0)

            # Determine label and console message based on confidence
            label_message = text[:20] + " with text" if conf > 0.625 else "diagram no text"

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


if __name__ == "__main__":
    # Example usage: run on one or more frames sequentially
    seg = Segmenter(use_server_mode=False)

    seg.segment_frame(
        "test_frames/00_23_40_00003.png",
        output_json="output_vis/frame1_regions.json",
        output_img="output_vis/frame1_vis.jpg"
    )

    seg.segment_frame(
        "test_frames/00_23_40_00003.png",
        output_json="output_vis/frame2_regions.json",
        output_img="output_vis/frame2_vis.jpg"
    )
