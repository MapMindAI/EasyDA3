# DA3 MVS Client (COLMAP + Triton + TSDF)

This project runs a DepthAnything3 Triton client on multi-view images using COLMAP sparse reconstruction for camera intrinsics/extrinsics, then optionally fuses predicted RGBD into a mesh with Open3D TSDF.

## Files

- `da3_client.py`: Triton client and full COLMAP-model inference entrypoint.
- `colmap_loader.py`: COLMAP sparse model I/O and chunking utilities.
- `fuse_rgbd_tsdf.py`: TSDF fusion script to build a mesh from RGB + depth + depth confidence.

## Requirements

- Python 3.9+
- `numpy`
- `opencv-python`
- `tritonclient[grpc]`
- `open3d` (only required for TSDF fusion)

Install example:

```bash
pip install numpy opencv-python tritonclient[grpc] open3d
```

## Input Data Layout

You need:

1. COLMAP sparse model directory (e.g. `.../sparse/0`) containing:
   - `images.bin` or `images.txt`
   - `cameras.bin` or `cameras.txt`
   - `points3D.bin` or `points3D.txt`
2. Image directory matching COLMAP `image.name` entries.

## Run DA3 Over Full COLMAP Model

```bash
python da3_client.py \
  --triton-url 0.0.0.0:8001 \
  --colmap-model-dir /path/to/sparse/0 \
  --image-dir /path/to/images \
  --output-dir /path/to/output \
  --chunk-size 10 \
  --min-shared-points 20 \
  --min-distance 0.02 \
  --min-rot 2.0
```

### What `da3_client.py` does

- Loads sparse model from COLMAP.
- Builds covisibility graph and chunks images.
- Applies movement constraints in chunking (`min_distance`, `min_rot`), with dynamic distance thresholding from global camera spacing.
- Converts chunk poses to first-image coordinate frame (`E0 = I`).
- Resizes input images and scales intrinsics to resized resolution.
- Sends `image`, `intrinsics_in`, `extrinsics_in` to Triton.
- Aligns output pose scale to input pose scale and rescales depth.
- Saves depth/conf arrays and visualizations.

## Output Structure

Under `--output-dir`:

- `depth/*.npy`: per-image depth map (`float32`)
- `depth_conf/*.npy`: per-image depth confidence (`float32`)
- `depth_pair_viz/*.png`: side-by-side depth/conf visualization

Filename mapping uses COLMAP image path stem with `/` and `\` replaced by `_`.

## Build Mesh with Open3D TSDF

```bash
python fuse_rgbd_tsdf.py \
  --colmap-model-dir /path/to/sparse/0 \
  --image-dir /path/to/images \
  --depth-dir /path/to/output/depth \
  --conf-dir /path/to/output/depth_conf \
  --output-mesh /path/to/output/mesh.ply \
  --conf-threshold 2.0
```

### TSDF Notes

- Depth pixels with confidence `< conf-threshold` are filtered out.
- Default `--conf-threshold` is `2.0`.
- You can tune:
  - `--voxel-length` (default `0.02`)
  - `--sdf-trunc` (default `0.08`)
  - `--depth-trunc` (default `20.0`)

## Quick Utility Test (Chunk Print)

To inspect COLMAP chunking:

```bash
python colmap_loader.py /path/to/sparse/0 --chunk-size 8 --min-shared-points 20
```

## Notes

- This repo assumes camera models are `PINHOLE` or `SIMPLE_PINHOLE` for intrinsic matrix conversion.
- If your Triton model uses different input tensor names, adjust:
  - `input_name`
  - `intrinsics_input_name`
  - `extrinsics_input_name`
  in `DepthAnything3(...)`.
