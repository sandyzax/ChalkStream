import cv2
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from core import RunConsole

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
    Content,
    ContentType,
    DiagramContent,
    draw_text_in_bbox,
)

EQUATION_OCR_URL = os.environ.get("CHALKSTREAM_EQUATION_OCR_URL", "http://127.0.0.1:8010/infer")
EQUATION_OCR_MODEL = os.environ.get("CHALKSTREAM_EQUATION_OCR_MODEL", "trained")
EQUATION_OCR_SEGMENT_INTERVAL = max(1, int(os.environ.get("CHALKSTREAM_EQUATION_OCR_SEGMENT_INTERVAL", "4")))
EQUATION_OCR_MIN_REGION_AGE = max(2, int(os.environ.get("CHALKSTREAM_EQUATION_OCR_MIN_REGION_AGE", "2")))
EQUATION_OCR_MIN_CONFIDENCE = float(os.environ.get("CHALKSTREAM_EQUATION_OCR_MIN_CONFIDENCE", "0.70"))
EQUATION_OCR_LABEL_OVERRIDE_CONFIDENCE = float(
    os.environ.get("CHALKSTREAM_EQUATION_OCR_LABEL_OVERRIDE_CONFIDENCE", "0.90")
)
EQUATION_OCR_TIMEOUT_S = float(os.environ.get("CHALKSTREAM_EQUATION_OCR_TIMEOUT_S", "120"))
EQUATION_OCR_MAX_GEN_LENGTH = max(8, int(os.environ.get("CHALKSTREAM_EQUATION_OCR_MAX_GEN_LENGTH", "128")))
EQUATION_OCR_NUM_BEAMS = min(8, max(1, int(os.environ.get("CHALKSTREAM_EQUATION_OCR_NUM_BEAMS", "1"))))
SEGMENT_INTERVAL_S = 1 / 2
PREVIEW_INTERVAL_S = 1 / 10
CONSOLE_SUMMARY_INTERVAL_SEGMENTS = max(1, int(os.environ.get("CHALKSTREAM_CONSOLE_SUMMARY_INTERVAL_SEGMENTS", "4")))
VERBOSE_REGION_OCR = os.environ.get("CHALKSTREAM_VERBOSE_REGION_OCR", "").strip().lower() in {"1", "true", "yes", "on"}
LABEL_PRIORITY = {
    "diagram": 0,
    "text": 1,
    "equation": 2,
}
REPO_ROOT = Path(__file__).resolve().parent.parent
SEGMENTATION_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "segmentation"
SEGMENTATION_FRAME_JSON = SEGMENTATION_ARTIFACT_DIR / "frame0.json"
SEGMENTATION_FRAME_VIS = SEGMENTATION_ARTIFACT_DIR / "frame0.jpg"


def _offset_polylines_to_global(polylines_local, x0: int, y0: int):
    out = []
    for poly in polylines_local:
        out_poly = []
        for pt in poly:
            out_poly.append([float(x0 + pt[0]), float(y0 + pt[1])])
        out.append(out_poly)
    return out


def _vectorize_region(crop, x0: int, y0: int):
    mask = mask_from_array(crop)
    skel = skeletonize_fast_skimage(mask)
    skel = prune_isolated_pixels(skel)

    polylines_yx, _nodes = trace_polylines(skel)
    polylines_local_xy = polylines_to_diagram_content(polylines_yx)
    return _offset_polylines_to_global(polylines_local_xy, x0=x0, y0=y0)


def _normalize_latex(latex):
    latex = str(latex or "").strip()
    if latex.startswith("$") and latex.endswith("$") and len(latex) >= 2:
        latex = latex[1:-1].strip()
    return latex


def _normalize_region_label(label):
    label = str(label or "diagram").lower()
    if label.startswith("equation"):
        return "equation"
    if label.startswith("text"):
        return "text"
    return "diagram"


def _parse_equation_ocr_confidence(confidence_value):
    if confidence_value is None:
        return None, None

    if isinstance(confidence_value, dict):
        score_value = confidence_value.get("score")
        if score_value is None:
            raise ValueError("Equation OCR confidence object is missing 'score'.")
        details = dict(confidence_value)
        details["score"] = float(score_value)
        return float(score_value), details

    return float(confidence_value), None


def _update_region_runtime_state(region, runtime_state, processing_clock_s):
    region_id = int(region.get("region_id", -1))
    state = runtime_state.get(region_id)
    if state is None:
        state = {
            "first_seen_processing_time_s": float(processing_clock_s),
            "last_processing_time_s": float(processing_clock_s),
            "last_raw_ocr_timestamp": region.get("last_ocr_time"),
            "last_ocr_processing_time_s": None,
        }
        runtime_state[region_id] = state

    state["last_processing_time_s"] = float(processing_clock_s)

    raw_last_ocr_time = region.get("last_ocr_time")
    if raw_last_ocr_time:
        if state.get("last_raw_ocr_timestamp") != raw_last_ocr_time:
            state["last_raw_ocr_timestamp"] = raw_last_ocr_time
            state["last_ocr_processing_time_s"] = float(processing_clock_s)
        elif state.get("last_ocr_processing_time_s") is None:
            state["last_ocr_processing_time_s"] = float(processing_clock_s)

    return state


def _build_region_metadata(region, content_type, processing_clock_s, runtime_state, equation_ocr_state=None, interaction_state=None):
    bbox = region.get("bbox")
    runtime = _update_region_runtime_state(region, runtime_state, processing_clock_s)
    base_label = str(region.get("display_label", region.get("label", "diagram")))
    classification = base_label
    effective_confidence = round(float(region.get("confidence", 0.0)), 3)
    classification_source = "lightweight_ocr"
    classification, classification_source = _resolve_effective_label_and_source(
        region,
        equation_ocr_state=equation_ocr_state,
        interaction_state=interaction_state,
    )
    if classification == "equation" and equation_ocr_state and equation_ocr_state.get("confidence") is not None:
        effective_confidence = round(float(equation_ocr_state.get("confidence")), 3)

    metadata = {
        "region_id": int(region.get("region_id", -1)),
        "classification": classification,
        "base_classification": base_label,
        "classification_source": classification_source,
        "classification_mode": _get_classification_mode(interaction_state),
        "region_label_override": _get_region_label_override(region, interaction_state),
        "raw_label": str(region.get("raw_label", region.get("label", "diagram"))),
        "content_type": content_type.name,
        "status": str(region.get("status", "existing")),
        "confidence": effective_confidence,
        "lightweight_confidence": round(float(region.get("confidence", 0.0)), 3),
        "age": int(region.get("age", 1)),
        "seen_count": int(region.get("seen_count", 1)),
        "missing_streak": int(region.get("missing_streak", 0)),
        "flip_streak": int(region.get("flip_streak", 0)),
        "bbox": [int(v) for v in bbox] if bbox is not None else None,
        "source_region_ids": [int(v) for v in region.get("source_region_ids", [])],
        "merge_applied": bool(region.get("source_region_ids")),
        "timestamp": str(region.get("timestamp", "")),
        "ocr_time_s": region.get("ocr_time_s"),
        "last_ocr_time": region.get("last_ocr_time"),
        "first_seen_time": region.get("first_seen_time"),
        "text_preview": str(region.get("text_preview", "")).strip()[:128],
        "segmentation_interval_s": SEGMENT_INTERVAL_S,
        "processing_time_s": runtime.get("last_processing_time_s"),
        "first_seen_processing_time_s": runtime.get("first_seen_processing_time_s"),
        "last_ocr_processing_time_s": runtime.get("last_ocr_processing_time_s"),
    }
    if equation_ocr_state:
        metadata.update({
            "equation_ocr_enabled": True,
            "equation_ocr_attempt_count": int(equation_ocr_state.get("attempt_count", 0)),
            "equation_ocr_last_attempt_segment": equation_ocr_state.get("last_attempt_segment"),
            "equation_ocr_last_attempt_time": equation_ocr_state.get("last_attempt_time"),
            "equation_ocr_success_count": int(equation_ocr_state.get("success_count", 0)),
            "equation_ocr_last_success_segment": equation_ocr_state.get("last_success_segment"),
            "equation_ocr_confidence": equation_ocr_state.get("confidence"),
            "equation_ocr_prediction_latex": equation_ocr_state.get("prediction_latex"),
            "equation_ocr_rendered": bool(equation_ocr_state.get("render_ready", False)),
            "equation_ocr_latency_ms": equation_ocr_state.get("latency_ms"),
            "equation_ocr_model": equation_ocr_state.get("model"),
            "equation_ocr_model_path": equation_ocr_state.get("model_path"),
            "equation_ocr_device": equation_ocr_state.get("device"),
            "equation_ocr_error": equation_ocr_state.get("last_error"),
            "equation_ocr_last_response_confidence": equation_ocr_state.get("last_response_confidence"),
            "equation_ocr_last_response_latex": equation_ocr_state.get("last_response_latex"),
            "equation_ocr_confidence_details": equation_ocr_state.get("confidence_details"),
            "equation_ocr_last_response_confidence_details": equation_ocr_state.get("last_response_confidence_details"),
        })
    return metadata


def _is_equation_region(region):
    label = str(region.get("display_label", region.get("label", "diagram"))).lower()
    return label.startswith("equation")


def _get_classification_mode(interaction_state=None):
    if interaction_state is None:
        return "normal"
    return str(interaction_state.get_classification_mode() or "normal").strip().lower()


def _get_region_label_override(region, interaction_state=None):
    if interaction_state is None:
        return None
    region_id = int(region.get("region_id", -1))
    override = interaction_state.get_region_label_override(region_id)
    return str(override).strip().lower() if override else None


def _resolve_effective_label_and_source(region, equation_ocr_state=None, interaction_state=None):
    classification_mode = _get_classification_mode(interaction_state)
    if classification_mode in {"diagram", "text", "equation"}:
        return classification_mode, "global_mode"

    region_override = _get_region_label_override(region, interaction_state)
    if region_override in {"diagram", "text", "equation"}:
        return region_override, "manual_override"

    if _has_equation_label_override(equation_ocr_state):
        return "equation", "equation_ocr"

    return str(region.get("display_label", region.get("label", "diagram"))).lower(), "lightweight_ocr"


def _is_equation_ocr_candidate(region, interaction_state=None):
    classification_mode = _get_classification_mode(interaction_state)
    if classification_mode == "equation":
        candidate = True
    elif classification_mode in {"diagram", "text"}:
        candidate = False
    else:
        region_override = _get_region_label_override(region, interaction_state)
        if region_override == "equation":
            candidate = True
        elif region_override in {"diagram", "text"}:
            candidate = False
        else:
            display_label = str(region.get("display_label", region.get("label", "diagram"))).lower()
            raw_label = str(region.get("raw_label", region.get("label", "diagram"))).lower()
            candidate = display_label in ("text", "equation") or raw_label in ("text", "equation")

    if not candidate:
        return False
    if str(region.get("status", "")).lower() == "missing":
        return False
    age = int(region.get("age", 1))
    seen_count = int(region.get("seen_count", 1))
    return age >= EQUATION_OCR_MIN_REGION_AGE or seen_count >= EQUATION_OCR_MIN_REGION_AGE


def _has_equation_label_override(equation_ocr_state):
    if not equation_ocr_state:
        return False
    prediction_latex = _normalize_latex(equation_ocr_state.get("prediction_latex"))
    confidence = equation_ocr_state.get("confidence")
    if not prediction_latex or confidence is None:
        return False
    return float(confidence) >= EQUATION_OCR_LABEL_OVERRIDE_CONFIDENCE


def _resolve_effective_label(region, equation_ocr_state, interaction_state=None):
    label, _source = _resolve_effective_label_and_source(
        region,
        equation_ocr_state=equation_ocr_state,
        interaction_state=interaction_state,
    )
    return label


def _union_bboxes(bboxes):
    x1 = min(int(bbox[0]) for bbox in bboxes)
    y1 = min(int(bbox[1]) for bbox in bboxes)
    x2 = max(int(bbox[0]) + int(bbox[2]) for bbox in bboxes)
    y2 = max(int(bbox[1]) + int(bbox[3]) for bbox in bboxes)
    return [x1, y1, x2 - x1, y2 - y1]


def _pick_representative_region(members):
    def representative_key(region):
        label = _normalize_region_label(region.get("display_label", region.get("label", "diagram")))
        status = str(region.get("status", "existing")).lower()
        return (
            1 if status != "missing" else 0,
            LABEL_PRIORITY.get(label, 0),
            float(region.get("confidence", 0.0)),
            int(region.get("age", 1)),
            int(region.get("seen_count", 1)),
        )

    return max(members, key=representative_key)


def _find_bbox_overlap_members(group_bbox, regions, used_ids):
    if not group_bbox:
        return []

    gx, gy, gw, gh = map(int, group_bbox)
    if gw <= 0 or gh <= 0:
        return []

    group_area = max(1, gw * gh)
    fallback_members = []
    for region in regions:
        region_id = int(region.get("region_id", -1))
        if region_id in used_ids:
            continue

        bbox = region.get("bbox")
        if not bbox:
            continue

        rx, ry, rw, rh = map(int, bbox)
        ix1 = max(gx, rx)
        iy1 = max(gy, ry)
        ix2 = min(gx + gw, rx + rw)
        iy2 = min(gy + gh, ry + rh)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter_area <= 0:
            continue

        region_area = max(1, rw * rh)
        overlap = inter_area / float(min(group_area, region_area))
        if overlap >= 0.25:
            fallback_members.append(region)

    return fallback_members


def _combine_text_preview(members):
    previews = []
    for member in members:
        preview = str(member.get("text_preview", "")).strip()
        if preview and preview not in previews:
            previews.append(preview)
    return " ".join(previews)[:128]


def _coalesce_regions_with_manual_merges(regions, merge_groups, interaction_state=None):
    if not merge_groups:
        return regions

    merged_regions = []
    used_source_ids = set()
    regions_by_id = {
        int(region.get("region_id", -1)): region
        for region in regions
        if region.get("region_id") is not None
    }

    for group in merge_groups:
        merged_region_id = int(group.get("merged_region_id", 0))
        source_region_ids = [int(source_id) for source_id in group.get("source_region_ids", []) if int(source_id) > 0]
        members = [regions_by_id[source_id] for source_id in source_region_ids if source_id in regions_by_id]
        if not members:
            members = _find_bbox_overlap_members(group.get("bbox"), regions, used_source_ids)
            source_region_ids = [int(member.get("region_id", -1)) for member in members if int(member.get("region_id", -1)) > 0]

        if not members:
            continue

        representative = _pick_representative_region(members)
        merged_bbox = _union_bboxes([member["bbox"] for member in members if member.get("bbox")])
        merged_status = "missing"
        if any(str(member.get("status", "existing")).lower() != "missing" for member in members):
            merged_status = "existing"

        label_source = _pick_representative_region(members)
        merged_label = _normalize_region_label(label_source.get("display_label", label_source.get("label", "diagram")))

        merged_region = dict(representative)
        merged_region["region_id"] = merged_region_id
        merged_region["bbox"] = merged_bbox
        merged_region["status"] = merged_status
        merged_region["raw_label"] = merged_label
        merged_region["display_label"] = merged_label
        merged_region["label"] = merged_label
        merged_region["confidence"] = float(label_source.get("confidence", representative.get("confidence", 0.0)))
        merged_region["text_preview"] = _combine_text_preview(members)
        merged_region["age"] = max(int(member.get("age", 1)) for member in members)
        merged_region["seen_count"] = max(int(member.get("seen_count", 1)) for member in members)
        merged_region["flip_streak"] = max(int(member.get("flip_streak", 0)) for member in members)
        merged_region["missing_streak"] = (
            0 if merged_status != "missing"
            else max(int(member.get("missing_streak", 0)) for member in members)
        )
        merged_region["source_region_ids"] = source_region_ids

        if interaction_state is not None:
            interaction_state.upsert_merge(merged_region_id, source_region_ids, bbox=merged_bbox)

        used_source_ids.update(source_region_ids)
        merged_regions.append(merged_region)

    passthrough_regions = []
    for region in regions:
        region_id = int(region.get("region_id", -1))
        if region_id not in used_source_ids:
            passthrough_regions.append(region)

    return merged_regions + passthrough_regions


def _equation_crop_to_png_bytes(crop):
    ok, buffer = cv2.imencode(".png", crop)
    if not ok:
        raise RuntimeError("Failed to encode equation crop as PNG.")
    return buffer.tobytes()


def _call_equation_ocr(crop):
    params = urllib.parse.urlencode({
        "model": EQUATION_OCR_MODEL,
        "include_confidence": "true",
        "max_gen_length": EQUATION_OCR_MAX_GEN_LENGTH,
        "num_beams": EQUATION_OCR_NUM_BEAMS,
    })
    url = f"{EQUATION_OCR_URL}?{params}"
    request = urllib.request.Request(
        url,
        data=_equation_crop_to_png_bytes(crop),
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=EQUATION_OCR_TIMEOUT_S) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _update_equation_ocr_state(warped, region, equation_ocr_state, segmentation_index, debug_console=None, interaction_state=None):
    region_id = int(region.get("region_id", -1))
    state = equation_ocr_state.get(region_id)
    if state is None:
        state = {
            "attempt_count": 0,
            "success_count": 0,
            "last_attempt_segment": None,
            "last_attempt_time": None,
            "last_success_segment": None,
            "prediction_latex": None,
            "confidence": None,
            "confidence_details": None,
            "last_response_confidence": None,
            "last_response_latex": None,
            "last_response_confidence_details": None,
            "latency_ms": None,
            "model": None,
            "model_path": None,
            "device": None,
            "last_error": None,
            "last_reported_error": None,
            "render_ready": False,
        }
        equation_ocr_state[region_id] = state

    should_attempt = _is_equation_ocr_candidate(region, interaction_state=interaction_state)
    if should_attempt:
        last_attempt_segment = state.get("last_attempt_segment")
        if last_attempt_segment is None or (segmentation_index - int(last_attempt_segment)) >= EQUATION_OCR_SEGMENT_INTERVAL:
            x, y, w, h = map(int, region.get("bbox", (0, 0, 0, 0)))
            crop = warped[y:y + h, x:x + w]
            state["attempt_count"] = int(state.get("attempt_count", 0)) + 1
            state["last_attempt_segment"] = segmentation_index
            state["last_attempt_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            try:
                response = _call_equation_ocr(crop)
                prediction_latex = _normalize_latex(response.get("prediction_latex"))
                confidence, confidence_details = _parse_equation_ocr_confidence(response.get("confidence"))
                state["last_response_latex"] = prediction_latex or None
                state["last_response_confidence"] = confidence
                state["last_response_confidence_details"] = confidence_details
                state["latency_ms"] = response.get("latency_ms")
                state["model"] = response.get("model")
                state["model_path"] = response.get("model_path")
                state["device"] = response.get("device")
                state["last_error"] = None
                state["last_reported_error"] = None
                if prediction_latex and confidence is not None and confidence >= EQUATION_OCR_MIN_CONFIDENCE:
                    state["success_count"] = int(state.get("success_count", 0)) + 1
                    state["last_success_segment"] = segmentation_index
                    best_confidence = state.get("confidence")
                    replaced_best = best_confidence is not None
                    if best_confidence is None or float(confidence) > float(best_confidence):
                        state["prediction_latex"] = prediction_latex
                        state["confidence"] = confidence
                        state["confidence_details"] = confidence_details
                        if debug_console is not None:
                            debug_console.equation_ocr_accept(region_id, confidence, prediction_latex, replaced_best=replaced_best)
                state["render_ready"] = bool(state.get("prediction_latex"))
            except Exception as exc:
                state["last_error"] = str(exc)
                if debug_console is not None and state.get("last_reported_error") != state["last_error"]:
                    debug_console.equation_ocr_error(region_id, state["last_error"])
                    state["last_reported_error"] = state["last_error"]
                state["render_ready"] = bool(state.get("prediction_latex"))

    return state


def _regions_to_contents(warped, regions, equation_ocr_state, segmentation_index, processing_clock_s, runtime_state, debug_console=None, interaction_state=None):
    contents = []

    for fallback_id, region in enumerate(regions):
        text_preview = str(region.get("text_preview", "")).strip()
        bbox = region.get("bbox")
        if bbox is None:
            continue

        x, y, w, h = map(int, bbox)
        if w <= 1 or h <= 1:
            continue
        if x < 0 or y < 0 or (x + w) > warped.shape[1] or (y + h) > warped.shape[0]:
            continue

        content_id = int(region.get("region_id", fallback_id))
        timestamp_str = str(region.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]))
        status = str(region.get("status", "existing"))
        confidence = float(region.get("confidence", 0.0))
        age = int(region.get("age", 1))
        equation_state = equation_ocr_state.get(content_id)

        if _is_equation_ocr_candidate(region, interaction_state=interaction_state):
            equation_state = _update_equation_ocr_state(
                warped,
                region,
                equation_ocr_state,
                segmentation_index,
                debug_console=debug_console,
                interaction_state=interaction_state,
            )

        label = _resolve_effective_label(region, equation_state, interaction_state=interaction_state)
        if label.startswith("equation") and equation_state and equation_state.get("confidence") is not None:
            confidence = float(equation_state.get("confidence"))

        if label == "text" and text_preview:
            content = Content(
                    id=content_id,
                    bbox=(x, y, w, h),
                    content_type=ContentType.WORDS,
                    confidence=confidence,
                    text_preview=text_preview[:128],
                    status=status,
                    timestamp=timestamp_str,
                    ocr_time_s=region.get("ocr_time_s"),
                    last_ocr_time=region.get("last_ocr_time"),
                    age=age,
                )
            content.region_metadata = _build_region_metadata(
                region,
                ContentType.WORDS,
                processing_clock_s,
                runtime_state,
                equation_state,
                interaction_state=interaction_state,
            )
            contents.append(content)
            continue

        content_type = ContentType.MATH if label.startswith("equation") else ContentType.DIAGRAM
        diagram = None
        latex_content = None

        if content_type == ContentType.MATH and equation_state and equation_state.get("render_ready"):
            latex_content = {
                "latex": equation_state.get("prediction_latex"),
                "confidence": equation_state.get("confidence"),
                "latency_ms": equation_state.get("latency_ms"),
                "model": equation_state.get("model"),
            }
        else:
            crop = warped[y:y + h, x:x + w]
            polylines_global_xy = _vectorize_region(crop, x0=x, y0=y)
            diagram = DiagramContent(polylines=polylines_global_xy)

        content = Content(
                id=content_id,
                bbox=(x, y, w, h),
                content_type=content_type,
                confidence=confidence,
                text_preview=text_preview[:128],
                status=status,
                timestamp=timestamp_str,
                diagram_content=diagram,
                latex_content=latex_content,
                age=age,
            )
        content.region_metadata = _build_region_metadata(
            region,
            content_type,
            processing_clock_s,
            runtime_state,
            equation_state,
            interaction_state=interaction_state,
        )
        contents.append(content)

    return contents


def _render_content_to_canvas_payload(content):
    x, y, w, h = map(int, content.bbox)
    if w <= 1 or h <= 1:
        return None

    if content.content_type == ContentType.MATH and content.latex_content:
        latex = _normalize_latex(content.latex_content.get("latex"))
        if latex:
            render_size = max(4, min(12, int(round(max(h, 1) / 12))))
            return {
                "id": int(content.id),
                "type": "latex",
                "content": latex,
                "size": render_size,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "metadata": getattr(content, "region_metadata", {}),
            }

    canvas_image = np.full((h, w, 3), 255, dtype=np.uint8)

    if content.content_type == ContentType.WORDS:
        draw_text_in_bbox(
            canvas_image,
            content.text_preview,
            (0, 0, w, h),
            height_fill=0.70,
            margin_frac=0.08,
            wrap=True,
        )
    else:
        polylines = []
        if content.diagram_content is not None:
            polylines = content.diagram_content.polylines

        for poly in polylines:
            if len(poly) < 2:
                continue

            local_pts = np.array(
                [[int(round(pt[0] - x)), int(round(pt[1] - y))] for pt in poly],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(canvas_image, [local_pts], isClosed=False, color=(0, 0, 0), thickness=2)

    return {
        "id": int(content.id),
        "type": "image",
        "content": canvas_image,
        "x": x,
        "y": y,
        "metadata": getattr(content, "region_metadata", {}),
    }


def _emit_canvas_updates(gui, contents, active_ids):
    current_ids = set()

    for content in contents:
        payload = _render_content_to_canvas_payload(content)
        if payload is None:
            continue
        current_ids.add(int(content.id))
        gui.load_content.emit(payload)

    for removed_id in sorted(active_ids - current_ids):
        gui.remove_content.emit(int(removed_id))

    return current_ids


def run_orchestrator(corners, gui=None, video_path: str = "pipeline/Test3.mp4", stop_event=None, show_debug=False, runtime_settings=None, interaction_state=None, debug_console=None):
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
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0

    # # Step 1: User selects board corners
    # print("[INFO] Select board corners...")
    # corners = select_board_corners(first_frame)
    # if corners is None:
    #     print("[INFO] Corner selection aborted.")
    #     return

    # Step 2: Initialize segmentation
    if debug_console is None:
        debug_console = RunConsole(
            enabled=True,
            summary_interval_segments=CONSOLE_SUMMARY_INTERVAL_SEGMENTS,
            verbose_ocr=VERBOSE_REGION_OCR,
        )

    seg = Segmenter(
        use_english_model=True,
        logger=debug_console,
        verbose_ocr=VERBOSE_REGION_OCR,
        log_saved_results=False,
    )
    SEGMENTATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2.1: Initialize noise_removal 
    first_warped = warp_board(first_frame, corners)
    noise_removal.previous_frame = first_warped.copy()
    noise_removal.avg_background = first_warped.astype(np.float32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    debug_console.startup(
        video_path=video_path,
        source_fps=source_fps,
        board_size=(first_warped.shape[1], first_warped.shape[0]),
        segment_interval_s=SEGMENT_INTERVAL_S,
        preview_interval_s=PREVIEW_INTERVAL_S,
        equation_ocr_url=EQUATION_OCR_URL,
        equation_ocr_model=EQUATION_OCR_MODEL,
    )

    frame_idx = 0
    last_seg_time = 0
    last_filter_time = 0
    last_filtered_frame = None
    last_vis = None
    active_ids = set()
    NOISE_INTERVAL = 1/10  # 10 FPS max
    SEGMENT_INTERVAL = SEGMENT_INTERVAL_S   # 2 FPS
    PREVIEW_INTERVAL = PREVIEW_INTERVAL_S
    segmentation_index = 0
    equation_ocr_state = {}
    region_runtime_state = {}
    last_preview_time = 0

    # Step 3: Process video frames
    while True:
        if stop_event is not None and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break
        
        # Rectify
        warped = warp_board(frame, corners)

        # Apply professor-filter logic and throttle professor filtering to interval
        now = time.time()
        settings_snapshot = runtime_settings.snapshot() if runtime_settings is not None else None
        if now - last_filter_time >= NOISE_INTERVAL:
            prof_filtered = noise_removal(warped, settings_snapshot=settings_snapshot)
            last_filtered_frame = prof_filtered
            last_filter_time = now
        else:
            # reuse last filtered frame to avoid blocking
            prof_filtered = last_filtered_frame

        # Throttle segmentation interval
        now = time.time()
        if now - last_seg_time >= SEGMENT_INTERVAL:
            seg.apply_runtime_settings(settings_snapshot)
            vis = seg.segment_array(
                prof_filtered,
                save_json=str(SEGMENTATION_FRAME_JSON),
                save_vis=str(SEGMENTATION_FRAME_VIS),
            )
            last_vis = vis
            segmentation_index += 1
            processing_clock_s = frame_idx / source_fps
            current_regions = getattr(seg, "prev_regions", []) or []
            merge_groups = interaction_state.snapshot_merges() if interaction_state is not None else []
            current_regions = _coalesce_regions_with_manual_merges(
                current_regions,
                merge_groups,
                interaction_state=interaction_state,
            )

            if gui is not None:
                contents = _regions_to_contents(
                    warped,
                    current_regions,
                    equation_ocr_state,
                    segmentation_index,
                    processing_clock_s,
                    region_runtime_state,
                    debug_console=debug_console,
                    interaction_state=interaction_state,
                )
                active_ids = _emit_canvas_updates(gui, contents, active_ids)
                stale_ids = set(equation_ocr_state) - active_ids
                for stale_id in stale_ids:
                    equation_ocr_state.pop(stale_id, None)
                stale_runtime_ids = set(region_runtime_state) - active_ids
                for stale_id in stale_runtime_ids:
                    region_runtime_state.pop(stale_id, None)

            debug_console.segment_summary(
                segmentation_index=segmentation_index,
                frame_idx=frame_idx,
                source_fps=source_fps,
                regions=current_regions,
                equation_ocr_state=equation_ocr_state,
                merge_groups=merge_groups,
            )

            last_seg_time = now
        else:
            vis = last_vis if last_vis is not None else prof_filtered

        if gui is not None and hasattr(gui, "preview_frame") and now - last_preview_time >= PREVIEW_INTERVAL:
            gui.preview_frame.emit({
                "mode": "original",
                "frame": warped.copy(),
            })
            filtered_preview = last_vis if last_vis is not None else prof_filtered
            if filtered_preview is not None:
                gui.preview_frame.emit({
                    "mode": "filtered",
                    "frame": filtered_preview.copy(),
                })
            last_preview_time = now

        # Step 4: Show visualization live
        if gui is None or show_debug:
            cv2.imshow("ChalkStream Segmentation", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        frame_idx += 1

    cap.release()
    if gui is None or show_debug:
        cv2.destroyAllWindows()
    debug_console.done(frame_idx, segmentation_index, source_fps)





def run1(video_path: any = 0):
    """
    Diagram-only test pipeline that works the same for:
      - Live capture (webcam / laptop camera): video_path is int (e.g., 0)
      - Video files (.mp4): video_path is str path

    Output:
      - Two resizable OpenCV windows
      - Writes JSON: artifacts/segmentation/frame0.json
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
    SEGMENTATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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
                out_json_path = str(SEGMENTATION_FRAME_JSON)
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
    print("[INFO] run1 done. Wrote artifacts/segmentation/frame0.json")
