# ChalkStream

Current source layout:

- `main.py`: application entrypoint
- `canvas/`: Qt GUI and source-selection flow
- `pipeline/`: orchestration and runtime pipeline, with videos in `pipeline/videos/`
- `preprocessing/`: board preprocessing and corner-selection helpers
- `segmentation/`: region detection, OCR-assisted segmentation, and board isolation
- `diagram_vectorization/`: diagram/vector content processing
- `core/`: shared runtime state, settings, and console reporting
- `assets/samples/`: small sample images used by local demos
- `scripts/`: ad hoc local utility/demo scripts
- `artifacts/`: generated run outputs such as segmentation JSON/visualizations

Notes:

- Runtime-generated files should go under `artifacts/`, not inside source directories.
- The video picker prefers `pipeline/videos/` but still supports the existing discovery flow.
