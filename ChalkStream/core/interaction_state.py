from copy import deepcopy
from threading import Lock


class InteractionStateStore:
    def __init__(self):
        self._lock = Lock()
        self._merge_groups = {}
        self._classification_mode = "normal"
        self._region_label_overrides = {}
        self._region_transforms = {}

    def snapshot_merges(self):
        with self._lock:
            return [deepcopy(self._merge_groups[key]) for key in sorted(self._merge_groups)]

    def upsert_merge(self, merged_region_id, source_region_ids, bbox=None, **extra_fields):
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
        with self._lock:
            group = self._merge_groups.pop(int(merged_region_id), None)
            return deepcopy(group) if group is not None else None

    def restore_merge(self, group):
        if not group:
            return None

        merged_region_id = int(group["merged_region_id"])
        restored_group = deepcopy(group)
        with self._lock:
            self._merge_groups[merged_region_id] = restored_group
        return deepcopy(restored_group)

    def clear(self):
        with self._lock:
            self._merge_groups.clear()
            self._classification_mode = "normal"
            self._region_label_overrides.clear()
            self._region_transforms.clear()

    def set_classification_mode(self, mode):
        normalized_mode = str(mode or "normal").strip().lower()
        if normalized_mode not in {"normal", "diagram", "text", "equation"}:
            raise ValueError(f"Unsupported classification mode: {mode}")
        with self._lock:
            self._classification_mode = normalized_mode
            return self._classification_mode

    def get_classification_mode(self):
        with self._lock:
            return self._classification_mode

    def set_region_label_override(self, region_id, label):
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
        with self._lock:
            return self._region_label_overrides.get(int(region_id))

    def snapshot_region_label_overrides(self):
        with self._lock:
            return deepcopy(self._region_label_overrides)

    def set_region_transform(self, region_id, x, y, scale):
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
        with self._lock:
            transform = self._region_transforms.get(int(region_id))
            return deepcopy(transform) if transform is not None else None

    def remove_region_transform(self, region_id):
        with self._lock:
            transform = self._region_transforms.pop(int(region_id), None)
            return deepcopy(transform) if transform is not None else None
