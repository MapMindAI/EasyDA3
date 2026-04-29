import argparse
import os

import cv2
import numpy as np
import open3d as o3d

from colmap_loader import load_sparse_model, get_intrinsics_matrix, get_extrinsics_matrix


def _safe_stem_from_colmap_name(image_name):
    stem = os.path.splitext(image_name)[0]
    return stem.replace("\\", "_").replace("/", "_")


def _load_depth_conf(depth_dir, conf_dir, image_name):
    safe_stem = _safe_stem_from_colmap_name(image_name)
    depth_path = os.path.join(depth_dir, f"{safe_stem}.npy")
    conf_path = os.path.join(conf_dir, f"{safe_stem}.npy")
    if not os.path.isfile(depth_path) or not os.path.isfile(conf_path):
        return None, None
    depth = np.load(depth_path).astype(np.float32)
    conf = np.load(conf_path).astype(np.float32)
    return depth, conf


def _build_o3d_intrinsic(K, width, height):
    return o3d.camera.PinholeCameraIntrinsic(
        int(width),
        int(height),
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )


def fuse_tsdf(
    colmap_model_dir,
    image_dir,
    depth_dir,
    conf_dir,
    output_mesh_path,
    conf_threshold=2.0,
    voxel_length=0.02,
    sdf_trunc=0.08,
    depth_trunc=20.0,
):
    images, cameras, _ = load_sparse_model(colmap_model_dir)
    image_ids = sorted(images.keys())

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    skipped_missing = 0
    skipped_invalid = 0

    for image_id in image_ids:
        image = images[image_id]
        camera = cameras[image.camera_id]

        rgb_path = os.path.join(image_dir, image.name)
        if not os.path.isfile(rgb_path):
            skipped_missing += 1
            continue

        depth, conf = _load_depth_conf(depth_dir, conf_dir, image.name)
        if depth is None:
            skipped_missing += 1
            continue

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            skipped_invalid += 1
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        if conf.shape != (h, w):
            conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

        # Filter low-confidence and invalid depth.
        valid = np.isfinite(depth) & np.isfinite(conf) & (conf >= float(conf_threshold)) & (depth > 0.0)
        depth_filtered = np.where(valid, depth, 0.0).astype(np.float32)
        if np.count_nonzero(depth_filtered) == 0:
            skipped_invalid += 1
            continue

        K = get_intrinsics_matrix(camera).astype(np.float64)
        # If source image resolution differs from COLMAP camera calibration resolution, rescale K.
        if camera.width > 0 and camera.height > 0 and (camera.width != w or camera.height != h):
            sx = float(w) / float(camera.width)
            sy = float(h) / float(camera.height)
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

        intrinsic = _build_o3d_intrinsic(K, w, h)
        extrinsic_w2c = get_extrinsics_matrix(image).astype(np.float64)

        color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_filtered)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, extrinsic_w2c)
        integrated += 1

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out_dir = os.path.dirname(output_mesh_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)

    print(f"Integrated frames: {integrated}")
    print(f"Skipped (missing files): {skipped_missing}")
    print(f"Skipped (invalid frames): {skipped_invalid}")
    print(f"Mesh written to: {output_mesh_path}")
    print(f"Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")


"""
python da3_mvs/fuse_rgbd_tsdf.py \
--colmap-model-dir ../EasyGaussianSplatting/data/gopro_test/sparse/0 \
--image-dir ../EasyGaussianSplatting/data/gopro_test/images \
--depth-dir ../EasyGaussianSplatting/data/gopro_test/da3/depth \
--conf-dir ../EasyGaussianSplatting/data/gopro_test/da3/depth_conf \
--output-mesh ../EasyGaussianSplatting/data/gopro_test/da3/mesh.ply \
--conf-threshold 2.0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse RGBD frames into a mesh with Open3D TSDF.")
    parser.add_argument("--colmap-model-dir", type=str, required=True, help="COLMAP sparse model directory")
    parser.add_argument("--image-dir", type=str, required=True, help="RGB image root directory")
    parser.add_argument("--depth-dir", type=str, required=True, help="Depth .npy directory")
    parser.add_argument("--conf-dir", type=str, required=True, help="Depth confidence .npy directory")
    parser.add_argument("--output-mesh", type=str, required=True, help="Output mesh path, e.g., out/mesh.ply")
    parser.add_argument("--conf-threshold", type=float, default=2.0, help="Filter depth where conf < threshold")
    parser.add_argument("--voxel-length", type=float, default=0.04, help="TSDF voxel length in scene units")
    parser.add_argument("--sdf-trunc", type=float, default=0.12, help="TSDF truncation distance")
    parser.add_argument("--depth-trunc", type=float, default=20.0, help="Depth truncation before TSDF integration")
    args = parser.parse_args()

    fuse_tsdf(
        colmap_model_dir=args.colmap_model_dir,
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        conf_dir=args.conf_dir,
        output_mesh_path=args.output_mesh,
        conf_threshold=args.conf_threshold,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
    )
