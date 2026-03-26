"""Persistent user interaction state that sits on top of the tracked board content."""

from copy import deepcopy
from threading import Lock


class InteractionStateStore:
    """Store manual overrides without coupling them directly to the pipeline state."""

    def __init__(self):
        self._lock = Lock()
        self._merge_groups = {}
        self._classification_mode = "normal"
        self._region_label_overrides = {}
        self._region_transforms = {}
        self._region_content_overrides = {}

    def snapshot_merges(self):
        """Return merge groups in a stable order for saving or replaying later."""
        with self._lock:
            return [deepcopy(self._merge_groups[key]) for key in sorted(self._merge_groups)]

    def upsert_merge(self, merged_region_id, source_region_ids, bbox=None, **extra_fields):
        """Create or replace a manual merge group for a synthetic region."""
        merged_region_id = int(merged_region_id)
        normalized_ids = []
        for source_id in source_region_ids or []:
            source_id = int(source_id)
            if source_id > 0 and source_id not in normalized_ids:
                normalized_ids.append(source_id)

        group = {
            "merged_region_id": merged_region_id,
            "source_region_ids": normalized_ids,
            "bbox": [int(v) for v in bbox] if bbox is not None else None,
        }
        if extra_fields:
            group.update(deepcopy(extra_fields))

        with self._lock:
            self._merge_groups[merged_region_id] = group
            return deepcopy(group)

    def remove_merge(self, merged_region_id):
        """Drop a manual merge group and return it for undo handling."""
        with self._lock:
            group = self._merge_groups.pop(int(merged_region_id), None)
            return deepcopy(group) if group is not None else None

    def restore_merge(self, group):
        """Restore a previously removed merge group."""
        if not group:
            return None

        merged_region_id = int(group["merged_region_id"])
        restored_group = deepcopy(group)
        with self._lock:
            self._merge_groups[merged_region_id] = restored_group
        return deepcopy(restored_group)

    def clear(self):
        """Reset all stored interaction state for a fresh session."""
        with self._lock:
            self._merge_groups.clear()
            self._classification_mode = "normal"
            self._region_label_overrides.clear()
            self._region_transforms.clear()
            self._region_content_overrides.clear()

    def set_classification_mode(self, mode):
        """Set the global classification mode used by the orchestrator."""
        normalized_mode = str(mode or "normal").strip().lower()
        if normalized_mode not in {"normal", "diagram", "text", "equation"}:
            raise ValueError(f"Unsupported classification mode: {mode}")
        with self._lock:
            self._classification_mode = normalized_mode
            return self._classification_mode

    def get_classification_mode(self):
        """Return the currently active global classification mode."""
        with self._lock:
            return self._classification_mode

    def set_region_label_override(self, region_id, label):
        """Store or clear a manual label override for one region."""
        region_id = int(region_id)
        normalized_label = str(label or "").strip().lower()
        if normalized_label not in {"", "diagram", "text", "equation"}:
            raise ValueError(f"Unsupported region label override: {label}")

        with self._lock:
            if not normalized_label:
                self._region_label_overrides.pop(region_id, None)
                return None
            self._region_label_overrides[region_id] = normalized_label
            return normalized_label

    def get_region_label_override(self, region_id):
        """Return the manual label override for a region, if one exists."""
        with self._lock:
            return self._region_label_overrides.get(int(region_id))

    def snapshot_region_label_overrides(self):
        """Return all manual label overrides for persistence."""
        with self._lock:
            return deepcopy(self._region_label_overrides)

    def snapshot_region_transforms(self):
        """Return all manual position and scale overrides."""
        with self._lock:
            return deepcopy(self._region_transforms)

    def set_region_transform(self, region_id, x, y, scale):
        """Record a user-driven move or resize for a single region."""
        region_id = int(region_id)
        transform = {
            "x": float(x),
            "y": float(y),
            "scale": float(scale),
        }
        with self._lock:
            self._region_transforms[region_id] = transform
            return deepcopy(transform)

    def get_region_transform(self, region_id):
        """Return the stored transform override for a region."""
        with self._lock:
            transform = self._region_transforms.get(int(region_id))
            return deepcopy(transform) if transform is not None else None

    def remove_region_transform(self, region_id):
        """Clear a stored transform override and return the removed value."""
        with self._lock:
            transform = self._region_transforms.pop(int(region_id), None)
            return deepcopy(transform) if transform is not None else None

    def set_region_content_override(self, region_id, override):
        """Store a visual content override for text, LaTeX, or image content."""
        region_id = int(region_id)
        if override is None:
            with self._lock:
                removed = self._region_content_overrides.pop(region_id, None)
                return deepcopy(removed) if removed is not None else None

        normalized = deepcopy(dict(override or {}))
        override_type = str(normalized.get("type") or "").strip().lower()
        if override_type not in {"text", "latex", "image"}:
            raise ValueError(f"Unsupported region content override type: {override_type}")

        normalized["type"] = override_type
        with self._lock:
            self._region_content_overrides[region_id] = normalized
            return deepcopy(normalized)

    def get_region_content_override(self, region_id):
        """Return the current visual override for a region, if one exists."""
        with self._lock:
            override = self._region_content_overrides.get(int(region_id))
            return deepcopy(override) if override is not None else None

    def remove_region_content_override(self, region_id):
        """Clear a visual content override and return the removed payload."""
        with self._lock:
            removed = self._region_content_overrides.pop(int(region_id), None)
            return deepcopy(removed) if removed is not None else None

    def snapshot_region_content_overrides(self):
        """Return all visual content overrides for persistence."""
        with self._lock:
            return deepcopy(self._region_content_overrides)
