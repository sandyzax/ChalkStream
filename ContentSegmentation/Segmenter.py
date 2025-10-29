import cv2
import numpy as np
import os
import json
import urllib.request
import tarfile
import time
from datetime import datetime
from paddleocr import TextRecognition

# Configurations

INPUT_PATH = "test_frames/00_23_40_00003.png"
OUTPUT_IMG_PATH = "output_vis/segmented_output.jpg"
OUTPUT_JSON_PATH = "output_vis/regions.json"

MIN_AREA = 800
MAX_AREA = 0.3
MORPH_KERNEL = (12, 12) # Bigger kernel more aggressive grouping for candidate regions
USE_SERVER_MODE = False  # True = high-accuracy, False = lightweight

# Model selection, download, and 
MODEL_BASE = os.path.join(os.getcwd(), "models", "rec")
os.makedirs(MODEL_BASE, exist_ok=True)

MODELS = {
    "mobile": {
        "name": "PP-OCRv5_mobile_rec_infer",
        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar",
    },
    "server": {
        "name": "PP-OCRv5_server_rec_infer",
        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar",
    },
}

MODE = "server" if USE_SERVER_MODE else "mobile"
MODEL_NAME = MODELS[MODE]["name"]
MODEL_URL = MODELS[MODE]["url"]
MODEL_PATH = os.path.join(MODEL_BASE, MODEL_NAME)

# Initialize the recognizer ONCE (global)
RECOGNIZER = None

def get_recognizer():
    global RECOGNIZER
    if RECOGNIZER is None:
        model_name = "PP-OCRv5_server_rec" if USE_SERVER_MODE else "PP-OCRv5_mobile_rec"
        print(f"[INFO] Loading {model_name} ...")
        RECOGNIZER = TextRecognition(model_name=model_name)
        print("[INFO] OCR model loaded.")
    return RECOGNIZER


def ensure_model(model_dir, url):
    """Download and extract model if missing."""
    # check for a key file from the archive
    if os.path.exists(os.path.join(model_dir, "inference.pdiparams")):
        print(f"[INFO] Model already present: {model_dir}")
        return
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    tar_path = os.path.join(os.path.dirname(model_dir), os.path.basename(url))
    print(f"[INFO] Downloading model from {url}...")
    urllib.request.urlretrieve(url, tar_path)
    print(f"[INFO] Extracting {tar_path}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=os.path.dirname(model_dir))
    os.remove(tar_path)
    print(f"[INFO] Model ready at {model_dir}")


ensure_model(MODEL_PATH, MODEL_URL)
print(f"[INFO] Using Recognition Model: {MODEL_NAME}")

# Component Functions
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grey scale
    blur = cv2.GaussianBlur(gray, (3, 3), 0) # Noise smoothing
    binary = cv2.adaptiveThreshold( # create black and white mask of chalk strokes and background
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL) # grouping rules / configs, bigger kernel more agressive grouping
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # infill back to original size
    return morph

def find_candidate_regions(binary, img_shape):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # congifuration for processing board mask into regions
    h, w = img_shape[:2]
    max_pixels = w * h 
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < MIN_AREA or area > MAX_AREA * max_pixels:
            continue
        if bw > w * 0.75:
            continue
        boxes.append((x, y, bw, bh))
    return boxes

def visualize_regions(img, boxes):
    vis = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis

def visualize_regions_with_labels(img, regions):
    # same as old, but adds the content labels and scores
    vis = img.copy()
    for region in regions:
        x, y, w, h = region["bbox"]
        label = region.get("label", "unknown")
        conf = region.get("confidence", 0.0)
        if label == "text":
            color = (0, 255, 0)
        elif label == "equation":
            color = (255, 0, 0)
        elif label == "diagram":
            color = (255, 0, 255)
        else:
            color = (200, 200, 200)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, f"{label} ({conf:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis

def save_results(data, out_json):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    if data and isinstance(data[0], tuple):
        results = [{"bbox": [int(x), int(y), int(w), int(h)]} for (x, y, w, h) in data]
    else:
        results = data
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} entries to {out_json}")

def score_regions(img, boxes):
    recognizer = get_recognizer() # if running live make sure to have 1 OCR model initialized for the session

    results = []
    for (x, y, w, h) in boxes:
        crop = img[y:y+h, x:x+w]
        tmp_path = "_tmp_crop.png"
        cv2.imwrite(tmp_path, crop)
        
        # Start timing 
        start_time = time.time()
        start_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[START] ({x}, {y}, {w}, {h}) @ {start_ts}")

        try:
            output = recognizer.predict(input=tmp_path)

            if not output:
                raise ValueError("empty output")
            result_dict = output[0] if isinstance(output[0], dict) else vars(output[0])
            text = result_dict.get("rec_text", "")
            confidence = result_dict.get("rec_score", 0.0)

        except Exception as e:
            print(f"OCR failed on region ({x}, {y}, {w}, {h}): {e}")
            text, confidence = "", 0.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # End timing 
        end_time = time.time()
        end_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        duration = round(end_time - start_time, 3)

        print(f"[END]   ({x}, {y}, {w}, {h}) @ {end_ts} | Duration: {duration:.3f}s")

        if any(sym in text for sym in ["=", "+", "-", "∫", "√", "π", "Σ"]):
            label = "equation"
        elif confidence > 0.65:
            label = "text"
        else:
            label = "diagram"

        results.append({
            "bbox": [x, y, w, h],
            "confidence": round(confidence, 3),
            "label": label,
            "text_preview": text
        })
    return results

# main function
def main():
    os.makedirs(os.path.dirname(OUTPUT_IMG_PATH), exist_ok=True) 
    img = load_image(INPUT_PATH)
    binary = preprocess(img) # extract binary from a processed image for candidate written region detection 
    boxes = find_candidate_regions(binary, img.shape)
    vis = visualize_regions(img, boxes) # put green region boxes around each written region
    cv2.imwrite(OUTPUT_IMG_PATH, vis)
    save_results(boxes, OUTPUT_JSON_PATH)
    print(f"Visualization saved to {OUTPUT_IMG_PATH}")

    scored = score_regions(img, boxes) # feed candidate regions through OCR and give set type (Text, Eq, Diagram)
    save_results(scored, OUTPUT_JSON_PATH)
    vis = visualize_regions_with_labels(img, scored)  # put scored region boxes around each written region
    cv2.imwrite(OUTPUT_IMG_PATH, vis)
    print(f"Annotated output saved to {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    main()
