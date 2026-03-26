"""Save and load editable board documents together with their image assets."""

import json
import re
from datetime import datetime
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parent.parent
SAVED_BOARDS_DIR = REPO_ROOT / "saved_boards"
BOARD_DOCUMENT_FILENAME = "board.json"
BOARD_ASSETS_DIRNAME = "assets"
BOARD_SCHEMA_VERSION = 1


def ensure_saved_boards_dir():
    """Create the saved board root directory if it is missing."""
    SAVED_BOARDS_DIR.mkdir(parents=True, exist_ok=True)
    return SAVED_BOARDS_DIR


def slugify_board_name(name):
    """Turn an arbitrary board title into a filesystem-safe folder name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "board").strip())
    text = text.strip("-.")
    return text or "board"


def discover_saved_boards():
    """Scan the saved board library and build picker entries for the GUI."""
    boards = []
    board_root = ensure_saved_boards_dir()

    for board_path in sorted(board_root.glob(f"*/{BOARD_DOCUMENT_FILENAME}")):
        try:
            with board_path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except Exception:
            continue

        board_name = str(document.get("board_name") or board_path.parent.name)
        board_size = dict(document.get("board_size") or {})
        width = int(board_size.get("width", 0) or 0)
        height = int(board_size.get("height", 0) or 0)
        saved_at = str(document.get("saved_at") or "")

        try:
            relative_path = str(board_path.relative_to(REPO_ROOT))
        except ValueError:
            relative_path = str(board_path)

        boards.append({
            "kind": "board",
            "value": str(board_path.resolve()),
            "name": board_name,
            "relative_path": relative_path,
            "saved_at": saved_at,
            "width": width,
            "height": height,
            "preferred": True,
            "size": board_path.stat().st_size,
        })

    boards.sort(key=lambda item: (item.get("saved_at", ""), item["name"].lower()), reverse=True)
    return boards


def build_interaction_state_snapshot(interaction_state):
    """Extract the interaction state fields that need to survive a save/load cycle."""
    if interaction_state is None:
        return {}

    return {
        "classification_mode": interaction_state.get_classification_mode(),
        "region_label_overrides": interaction_state.snapshot_region_label_overrides(),
        "region_transforms": interaction_state.snapshot_region_transforms(),
    }


def apply_interaction_state_snapshot(interaction_state, snapshot):
    """Replay a saved interaction-state snapshot into a fresh store."""
    if interaction_state is None:
        return

    interaction_state.clear()
    snapshot = dict(snapshot or {})

    classification_mode = str(snapshot.get("classification_mode") or "normal")
    interaction_state.set_classification_mode(classification_mode)

    for region_id, label in dict(snapshot.get("region_label_overrides") or {}).items():
        interaction_state.set_region_label_override(int(region_id), label)

    for region_id, transform in dict(snapshot.get("region_transforms") or {}).items():
        transform = dict(transform or {})
        interaction_state.set_region_transform(
            int(region_id),
            float(transform.get("x", 0.0)),
            float(transform.get("y", 0.0)),
            float(transform.get("scale", 1.0)),
        )


def _normalize_snapshot_for_save(snapshot, board_dir):
    """Split item snapshots into JSON data plus image assets on disk."""
    entry = dict(snapshot)
    item_type = str(entry.get("type") or "").strip().lower()
    content = entry.pop("content", None)

    if item_type == "image":
        asset_rel_path = Path(BOARD_ASSETS_DIRNAME) / f"item_{int(entry['id'])}.png"
        asset_path = board_dir / asset_rel_path
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        if content is None or not cv2.imwrite(str(asset_path), content):
            raise RuntimeError(f"Failed to save board image asset for item {entry['id']}.")
        entry["asset_path"] = str(asset_rel_path).replace("\\", "/")
    else:
        entry["content"] = content

    return entry


def _save_optional_image_asset(board_dir, image, filename):
    """Write an optional image asset and return its relative asset path."""
    if image is None:
        return None

    asset_rel_path = Path(BOARD_ASSETS_DIRNAME) / filename
    asset_path = board_dir / asset_rel_path
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(asset_path), image):
        raise RuntimeError(f"Failed to save board image asset: {asset_path}")
    return str(asset_rel_path).replace("\\", "/")


def save_board_document(
    board_name,
    width,
    height,
    item_snapshots,
    interaction_state_snapshot,
    source_info=None,
    selected_view_keys=None,
    existing_board_path=None,
    original_frame=None,
):
    """Write a complete board document and return the path to ``board.json``."""
    ensure_saved_boards_dir()

    if existing_board_path:
        board_path = Path(existing_board_path).resolve()
        board_dir = board_path.parent
        board_name = str(board_name or board_dir.name)
    else:
        slug = slugify_board_name(board_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        board_dir = SAVED_BOARDS_DIR / f"{slug}_{timestamp}"
        board_path = board_dir / BOARD_DOCUMENT_FILENAME

    board_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = board_dir / BOARD_ASSETS_DIRNAME
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Keep the save folder clean so removed items do not leave stale image assets behind.
    for stale_asset in assets_dir.glob("*"):
        if stale_asset.is_file():
            stale_asset.unlink()

    items = [
        _normalize_snapshot_for_save(snapshot, board_dir)
        for snapshot in item_snapshots
    ]
    original_frame_asset_path = _save_optional_image_asset(board_dir, original_frame, "original_frame.png")

    document = {
        "schema_version": BOARD_SCHEMA_VERSION,
        "board_name": str(board_name or board_dir.name),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "board_size": {
            "width": int(width),
            "height": int(height),
        },
        "source_info": dict(source_info or {}),
        "selected_view_keys": list(selected_view_keys or ["canvas"]),
        "interaction_state": dict(interaction_state_snapshot or {}),
        "original_frame_asset_path": original_frame_asset_path,
        "items": items,
    }

    with board_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)

    return board_path


def load_board_document(board_path):
    """Load a saved board document and materialize any referenced image assets."""
    board_path = Path(board_path).resolve()
    with board_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    board_dir = board_path.parent
    original_frame = None
    original_frame_asset_path = document.get("original_frame_asset_path")
    if original_frame_asset_path:
        asset_path = board_dir / str(original_frame_asset_path)
        original_frame = cv2.imread(str(asset_path))
        if original_frame is None:
            raise FileNotFoundError(f"Saved board original frame asset not found: {asset_path}")

    items = []
    for stored_item in list(document.get("items") or []):
        item = dict(stored_item)
        item_type = str(item.get("type") or "").strip().lower()
        if item_type == "image":
            asset_rel_path = item.pop("asset_path", None)
            if not asset_rel_path:
                raise FileNotFoundError(f"Saved board image asset path missing for item {item.get('id')}.")
            asset_path = board_dir / asset_rel_path
            image = cv2.imread(str(asset_path))
            if image is None:
                raise FileNotFoundError(f"Saved board image asset not found: {asset_path}")
            item["content"] = image
        items.append(item)

    board_size = dict(document.get("board_size") or {})
    return {
        "board_path": str(board_path),
        "board_name": str(document.get("board_name") or board_dir.name),
        "width": int(board_size.get("width", 0) or 0),
        "height": int(board_size.get("height", 0) or 0),
        "items": items,
        "interaction_state": dict(document.get("interaction_state") or {}),
        "source_info": dict(document.get("source_info") or {}),
        "selected_view_keys": list(document.get("selected_view_keys") or ["canvas"]),
        "original_frame": original_frame,
    }
