# DA3 SLAM (Simple Guide)

This module runs a hybrid SLAM pipeline:
- Optical flow frontend for realtime tracking and keyframe selection.
- DepthAnything3 (DA3) backend for delayed depth/pose refinement.

## Requirements

From repo root:

```bash
pip install -r requirement.txt
```

Start Triton server (from EasyTensorRT project in https://github.com/MapMindAI/EasyTensorRT/tree/ly_da3_pair):

```bash
./run_server_trt.sh
```

## Pipelines

Two pipeline modes are available:

1. `build_default_pipeline()`
- DA3 runs on a 10-keyframe chunk.
- Triggered every N new keyframes (default stride in code).
- Output dir: `output_da3_map`.

2. `build_sequential_pair_pipeline()`
- DA3 runs on each sequential keyframe pair `[k-1, k]`.
- A new DA3 backend job is scheduled whenever optical flow promotes a keyframe.
- Output dir: `output_da3_map_seq_pair`.

## Run

Default demo entry:

```bash
python da3_slam/da3_streaming.py
```

## Test Data

- AmsterdamMorningDrive test images:
  - https://drive.google.com/file/d/1hQWkxjgtKRgGxvAbxjnFFq6jxW2QFkl0/view?usp=drive_link

If you want pair mode, change the pipeline construction in `main()`:

```python
pipeline = build_sequential_pair_pipeline()
```

https://github.com/user-attachments/assets/f1f491ab-0dff-4e5b-be26-8ef0510b5a17

## Open3D Viewer

- Camera frustum rendering uses SLAM poses directly.
- Follow-camera orientation is aligned from SLAM camera axes (`front=+Z`, `up=-Y`) to avoid the initial 180-degree virtual camera flip.
- Keys:
  - `F`: toggle follow latest pose
  - `M`: toggle chunk TSDF mesh
  - `P`: toggle chunk TSDF point cloud
