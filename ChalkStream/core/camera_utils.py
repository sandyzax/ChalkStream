"""Helpers for opening cameras and negotiating a usable capture resolution."""

import os

import cv2


CAMERA_RESOLUTION_OPTIONS = (
    ("max", "Max Available", None),
    ("3840x2160", "3840 x 2160", (3840, 2160)),
    ("2560x1440", "2560 x 1440", (2560, 1440)),
    ("1920x1080", "1920 x 1080", (1920, 1080)),
    ("1600x900", "1600 x 900", (1600, 900)),
    ("1280x720", "1280 x 720", (1280, 720)),
    ("1024x768", "1024 x 768", (1024, 768)),
    ("800x600", "800 x 600", (800, 600)),
    ("640x480", "640 x 480", (640, 480)),
)

DEFAULT_CAMERA_RESOLUTION_OPTION = "max"


def normalize_camera_resolution_option(value):
    """Collapse user-facing resolution input into one supported option key."""
    normalized = str(value or DEFAULT_CAMERA_RESOLUTION_OPTION).strip().lower()
    for option_key, _label, _dims in CAMERA_RESOLUTION_OPTIONS:
        if option_key == normalized:
            return option_key
    return DEFAULT_CAMERA_RESOLUTION_OPTION


def camera_resolution_label(option_key):
    """Return the human-readable label for a stored camera resolution option."""
    normalized = normalize_camera_resolution_option(option_key)
    for candidate_key, label, _dims in CAMERA_RESOLUTION_OPTIONS:
        if candidate_key == normalized:
            return label
    return "Max Available"


def camera_resolution_dimensions(option_key):
    """Return the width and height pair for a stored option, if it has one."""
    normalized = normalize_camera_resolution_option(option_key)
    for candidate_key, _label, dims in CAMERA_RESOLUTION_OPTIONS:
        if candidate_key == normalized:
            return dims
    return None


def open_camera_capture(camera_index):
    """Open a camera with the most reliable backend available on this platform."""
    if os.name == "nt":
        capture = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(int(camera_index))


def _read_camera_frame(cap):
    """Read one frame and collapse failed reads to ``None``."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _apply_camera_resolution_request(cap, width, height):
    """Ask the device for a specific resolution and report what it actually delivered."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    frame = _read_camera_frame(cap)
    if frame is None:
        return None
    actual_width = int(frame.shape[1])
    actual_height = int(frame.shape[0])
    return {
        "requested_width": int(width),
        "requested_height": int(height),
        "actual_width": actual_width,
        "actual_height": actual_height,
        "frame": frame,
    }


def open_configured_camera_capture(camera_index, resolution_option=DEFAULT_CAMERA_RESOLUTION_OPTION):
    """Open a camera and negotiate either a requested mode or the best available one."""
    cap = open_camera_capture(camera_index)
    info = {
        "camera_index": int(camera_index),
        "requested_option": normalize_camera_resolution_option(resolution_option),
        "requested_width": None,
        "requested_height": None,
        "actual_width": None,
        "actual_height": None,
    }
    if not cap.isOpened():
        return cap, info, None

    # MJPG is commonly the format that allows USB webcams to expose their higher modes.
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    normalized_option = normalize_camera_resolution_option(resolution_option)
    chosen_result = None

    if normalized_option == "max":
        best_area = -1
        best_dims = None
        for option_key, _label, dims in CAMERA_RESOLUTION_OPTIONS:
            if option_key == "max" or dims is None:
                continue
            result = _apply_camera_resolution_request(cap, dims[0], dims[1])
            if result is None:
                continue
            area = int(result["actual_width"]) * int(result["actual_height"])
            if area > best_area:
                best_area = area
                best_dims = (int(result["requested_width"]), int(result["requested_height"]))
                chosen_result = result
            if result["actual_width"] >= dims[0] and result["actual_height"] >= dims[1]:
                best_dims = (int(result["requested_width"]), int(result["requested_height"]))
                chosen_result = result
                break

        if best_dims is not None:
            confirmed = _apply_camera_resolution_request(cap, best_dims[0], best_dims[1])
            if confirmed is not None:
                chosen_result = confirmed
        elif chosen_result is None:
            frame = _read_camera_frame(cap)
            if frame is not None:
                chosen_result = {
                    "requested_width": None,
                    "requested_height": None,
                    "actual_width": int(frame.shape[1]),
                    "actual_height": int(frame.shape[0]),
                    "frame": frame,
                }
    else:
        dims = camera_resolution_dimensions(normalized_option)
        if dims is not None:
            chosen_result = _apply_camera_resolution_request(cap, dims[0], dims[1])
        if chosen_result is None:
            frame = _read_camera_frame(cap)
            if frame is not None:
                chosen_result = {
                    "requested_width": dims[0] if dims else None,
                    "requested_height": dims[1] if dims else None,
                    "actual_width": int(frame.shape[1]),
                    "actual_height": int(frame.shape[0]),
                    "frame": frame,
                }

    if chosen_result is not None:
        info.update({
            "requested_width": chosen_result.get("requested_width"),
            "requested_height": chosen_result.get("requested_height"),
            "actual_width": chosen_result.get("actual_width"),
            "actual_height": chosen_result.get("actual_height"),
        })
        return cap, info, chosen_result.get("frame")

    return cap, info, None
