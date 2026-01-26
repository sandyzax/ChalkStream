from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import cv2
import math


class ContentType(Enum):
    WORDS = 1
    MATH = 2
    DIAGRAM = 3
    IMAGE = 4

@dataclass
class DiagramContent:
    polylines: List[List[List[float]]]  # [[[x,y], [x,y], ...], ...]

    def to_dict(self):
        return {"polylines": self.polylines}
    
@dataclass
class Content:
    """
    Canonical content object shared across pipeline stages.
    """
    id: int
    bbox: tuple            #(x,y,w,h)
    content_type: ContentType
    confidence: float
    text_preview: str
    status: str
    timestamp: str

     # Optional fields
    ocr_time_s: Optional[float] = None
    last_ocr_time: Optional[float] = None
    age: Optional[int] = None

    diagram_content: Optional[DiagramContent] = None
    latex_content: Optional[Dict[str, Any]] = None  # later

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "content_type": self.content_type.value,
            "confidence": self.confidence,
            "text_preview": self.text_preview,
            "status": self.status,
            "timestamp": self.timestamp,
            "ocr_time_s": self.ocr_time_s,
            "last_ocr_time": self.last_ocr_time,
            "age": self.age,
            "diagram_content": None if self.diagram_content is None else self.diagram_content.to_dict(),
            "latex_content": self.latex_content,
        }

    def to_dict(self):
        d = {
            "id": self.id,
            "bbox": list(self.bbox),
            "content_type": self.content_type.value,
            "confidence": self.confidence,
            "text_preview": self.text_preview,
            "status": self.status,
            "timestamp": self.timestamp,
        }
        if self.diagram_content:
            d["diagram_content"] = self.diagram_content.to_dict()
        return d
    
class ContentLoad:
    id_count:int
    contents:list[Content] = []
    board_size:tuple 
    bg_color:tuple #
    default_lifespan:int = 10

    def __init__(self,size:tuple,bg_color:tuple = (255,255,255)) -> None:
        self.id_count = 0
        self.board_size = size
        self.bg_color = bg_color


    
    def intersection_area(self, a, b)->float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def overlap_percentage(self, a, b)->float:
        inter = self.intersection_area(a, b)
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]

        return inter / min(area_a, area_b)

    def update_frame(self,new_contents:list[Content])->None:
        updated_contents:list = []
        for new_content in new_contents:
            new = True
            for content in self.contents[:]:
                print(333)
                print(self.overlap_percentage(new_content.bbox,content.bbox))
                if new_content.contentType == content.contentType and self.overlap_percentage(new_content.bbox,content.bbox) > 0.6:
                    print(111)
                    content.bbox = content.bbox[:2] + new_content.bbox[2:]
                    content.content = new_content.content
                    content.life_span = self.default_lifespan
                    updated_contents.append(content)
                    self.contents.remove(content)
                    new = False
                    break
            if new:
                updated_contents.append(Content(self.id_count,new_content.contentType,new_content.bbox,new_content.content,self.default_lifespan))
                self.id_count = self.id_count + 1
        for content in self.contents:
            content.life_span -= 1
            if content.life_span > 0:
                updated_contents.append(content)
        print("xxx ",len(updated_contents))
        self.contents = updated_contents

    
    def create_content(self,content,content_type:ContentType,position:tuple)->Content:
        h, w = content.shape[:2]

        return Content(-1, content_type,(position[0],position[1],w,h),content,self.default_lifespan)

    def display_content(self):
        background = np.full((self.board_size[1], self.board_size[0], 3), self.bg_color, dtype=np.uint8)
        for content in self.contents:
            img = None
            if content.contentType == ContentType.IMAGE:
                img = content.content
            else:
                print(f"Content Type {content.contentType} Not Supported (yet), skipping")
                continue
            x,y,w,h = content.bbox

            background[y:y+h, x:x+w] = img
        cv2.imshow("Result", background)
        cv2.waitKey(0)

def draw_text_in_bbox(
        canvas,
        text: str,
        bbox,                       # (x,y,w,h)
        *,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        height_fill=0.70,           # % of bbox height to target for font height
        margin_frac=0.08,           # % of bbox min(w,h) for padding
        min_scale=0.25,
        max_scale=3.0,
        thickness_at_scale=1.2,     # thickness ≈ thickness_at_scale * scale
        line_spacing=1.15,          # multiplier on line height
        wrap=True,
    ):
        x, y, w, h = map(int, bbox)
        if w <= 2 or h <= 2:
            return

        text = (text or "").strip()
        if not text:
            return

        # Padding
        pad = int(max(1, margin_frac * min(w, h)))
        x0, y0 = x + pad, y + pad
        x1, y1 = x + w - pad, y + h - pad
        avail_w = max(1, x1 - x0)
        avail_h = max(1, y1 - y0)

        # ---- 1) Start with a scale based on bbox height ----
        # OpenCV text height is not 1:1 with scale; we estimate from a probe.
        probe = "Ag"
        (probe_w, probe_h), probe_base = cv2.getTextSize(probe, font, 1.0, 1)
        target_h_px = max(1, int(avail_h * height_fill))
        scale_h = target_h_px / max(1, probe_h)

        # ---- 2) Also constrain by width (string length) ----
        # Measure the actual string width at scale 1 and shrink/grow to fit.
        (tw1, th1), base1 = cv2.getTextSize(text, font, 1.0, 1)
        scale_w = avail_w / max(1, tw1)

        # Choose the tighter of the two so it fits both height + width
        scale = max(min_scale, min(max_scale, min(scale_h, scale_w)))

        # Thickness linked to scale
        thickness = max(1, int(round(thickness_at_scale * scale)))

        def measure(s, sc):
            (tw, th), base = cv2.getTextSize(s, font, sc, thickness)
            return tw, th, base

        # ---- 3) Optional wrapping if it still won't fit or if it's long ----
        lines = [text]
        if wrap:
            # Greedy word-wrap to fit width
            words = text.split()
            if len(words) >= 2:
                lines = []
                cur = words[0]
                for w_i in words[1:]:
                    candidate = cur + " " + w_i
                    cand_w, _, _ = measure(candidate, scale)
                    if cand_w <= avail_w:
                        cur = candidate
                    else:
                        lines.append(cur)
                        cur = w_i
                lines.append(cur)

            # If still too wide (e.g. one very long token), fall back to char wrapping
            fixed = []
            for ln in lines:
                ln_w, _, _ = measure(ln, scale)
                if ln_w <= avail_w:
                    fixed.append(ln)
                    continue
                # char wrap
                chunk = ""
                for ch in ln:
                    cand = chunk + ch
                    cand_w, _, _ = measure(cand, scale)
                    if cand_w <= avail_w or not chunk:
                        chunk = cand
                    else:
                        fixed.append(chunk)
                        chunk = ch
                if chunk:
                    fixed.append(chunk)
            lines = fixed

        # ---- 4) If wrapped into multiple lines, ensure total height fits ----
        _, line_h, line_base = measure("Ag", scale)
        step = int(math.ceil(line_h * line_spacing))
        total_h = step * len(lines)

        if total_h > avail_h:
            # shrink scale to make vertical fit
            shrink = avail_h / max(1, total_h)
            scale = max(min_scale, scale * shrink)
            thickness = max(1, int(round(thickness_at_scale * scale)))
            _, line_h, _ = measure("Ag", scale)
            step = int(math.ceil(line_h * line_spacing))

        # ---- 5) Draw, top-left anchored inside bbox ----
        # Put first baseline a bit below the top padding.
        baseline_y = y0 + line_h
        for i, ln in enumerate(lines):
            yy = baseline_y + i * step
            if yy > y1:  # stop if out of box
                break
            cv2.putText(canvas, ln, (x0, yy), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

def test1():
    overlay = cv2.imread("input.png")
    content_load = ContentLoad((1200,800))
    overlay = content_load.create_content(overlay,ContentType.IMAGE,(50,100))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    content_load.display_content()

def test2():
    img = cv2.imread("input.png")
    content_load = ContentLoad((1200,800))
    overlay = content_load.create_content(img,ContentType.IMAGE,(50,100))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    overlay = content_load.create_content(img,ContentType.IMAGE,(60,120))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    content_load.display_content()

def test3():
    img = cv2.imread("input.png")
    content_load = ContentLoad((1200,800))
    overlay = content_load.create_content(img,ContentType.IMAGE,(50,100))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    overlay = content_load.create_content(img,ContentType.IMAGE,(300,140))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    content_load.display_content()

if __name__ == "__main__":
    test1()
    # test2()
    # test3()

    

                    







