from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d


def build_open3d_tsdf_from_saved_keyframes(
    map_dir: str | Path,
    output_mesh_path: Optional[str | Path] = None,
    output_pcd_path: Optional[str | Path] = None,
    use_chunks: Optional[Sequence[int]] = None,
    integrate_each_keyframe_once: bool = True,
    voxel_length: float = 0.03,
    sdf_trunc: float = 0.12,
    depth_trunc: float = 80.0,
    depth_scale: float = 1.0,
    volume_unit_resolution: int = 16,
    depth_sampling_stride: int = 1,
    min_depth: float = 1e-4,
    use_pose_w2c_if_available: bool = True,
    verbose: bool = True,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, Dict]:
    """
    Build an Open3D TSDF map from saved DA3-SLAM keyframes.

    Args:
        map_dir:
            Path to output_da3_map directory containing manifest.json.

        output_mesh_path:
            Optional path to save extracted mesh, for example:
                "map_mesh.ply"

        output_pcd_path:
            Optional path to save extracted point cloud, for example:
                "map_cloud.ply"

        use_chunks:
            Optional list of chunk IDs to fuse.
            If None, uses all chunks listed in manifest.json.

        integrate_each_keyframe_once:
            If True, avoids duplicate integration of overlapping keyframes.
            Recommended because DA3 chunks overlap.

        voxel_length:
            TSDF voxel size. Smaller gives more detail but uses more memory.

        sdf_trunc:
            TSDF truncation distance. Usually 3x to 5x voxel_length.

        depth_trunc:
            Ignore depth values farther than this.

        depth_scale:
            Open3D divides the raw depth image by depth_scale.
            Since your saved DA3 depth is float depth already, use 1.0.

        volume_unit_resolution:
            Open3D ScalableTSDFVolume block resolution.

        depth_sampling_stride:
            Open3D TSDF integration depth sampling stride.

        min_depth:
            Depth values <= min_depth are invalidated before integration.

        use_pose_w2c_if_available:
            Open3D integration uses extrinsic. In this pipeline we save both
            pose_c2w and pose_w2c. Use pose_w2c directly if available.

        verbose:
            Print progress.

    Returns:
        mesh:
            Extracted triangle mesh.

        pcd:
            Extracted point cloud.

        stats:
            Dictionary with integration statistics.
    """

    map_dir = Path(map_dir)
    manifest_path = map_dir / "manifest.json"
    keyframe_dir = map_dir / "keyframes"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Cannot find manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    selected_keyframe_ids = _collect_keyframes_from_manifest_chunks(
        manifest=manifest,
        use_chunks=use_chunks,
        integrate_each_keyframe_once=integrate_each_keyframe_once,
    )

    if verbose:
        print(f"[TSDF] selected keyframes: {len(selected_keyframe_ids)}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        volume_unit_resolution=int(volume_unit_resolution),
        depth_sampling_stride=int(depth_sampling_stride),
    )

    integrated_ids = []
    skipped = []

    for image_id in selected_keyframe_ids:
        try:
            color, depth, intrinsic, extrinsic_w2c = _load_keyframe_rgbd_camera(
                keyframe_dir=keyframe_dir,
                image_id=image_id,
                min_depth=min_depth,
                depth_trunc=depth_trunc,
                use_pose_w2c_if_available=use_pose_w2c_if_available,
            )
        except Exception as e:
            skipped.append((image_id, str(e)))
            if verbose:
                print(f"[TSDF][skip] kf {image_id}: {e}")
            continue

        h, w = depth.shape[:2]

        o3d_color = o3d.geometry.Image(color)
        o3d_depth = o3d.geometry.Image(depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d_color,
            depth=o3d_depth,
            depth_scale=float(depth_scale),
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )

        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])

        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=int(w),
            height=int(h),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        volume.integrate(
            rgbd,
            o3d_intrinsic,
            extrinsic_w2c.astype(np.float64),
        )

        integrated_ids.append(image_id)

        if verbose and len(integrated_ids) % 10 == 0:
            print(f"[TSDF] integrated {len(integrated_ids)} frames")

    if verbose:
        print(f"[TSDF] extracting mesh and point cloud...")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    pcd = volume.extract_point_cloud()

    if output_mesh_path is not None:
        output_mesh_path = Path(output_mesh_path)
        output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output_mesh_path), mesh)
        if verbose:
            print(f"[TSDF] saved mesh: {output_mesh_path}")

    if output_pcd_path is not None:
        output_pcd_path = Path(output_pcd_path)
        output_pcd_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_pcd_path), pcd)
        if verbose:
            print(f"[TSDF] saved point cloud: {output_pcd_path}")

    stats = {
        "map_dir": str(map_dir),
        "num_selected_keyframes": len(selected_keyframe_ids),
        "num_integrated": len(integrated_ids),
        "num_skipped": len(skipped),
        "integrated_ids": integrated_ids,
        "skipped": skipped,
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
    }

    return mesh, pcd, stats


def _collect_keyframes_from_manifest_chunks(
    manifest: Dict,
    use_chunks: Optional[Sequence[int]],
    integrate_each_keyframe_once: bool,
) -> List[int]:
    """
    Collect image IDs from manifest chunks.

    If chunks overlap, duplicated image IDs are removed by default.
    """

    chunks = manifest.get("chunks", [])

    if use_chunks is not None:
        use_chunks = set(int(x) for x in use_chunks)
        chunks = [c for c in chunks if int(c["chunk_id"]) in use_chunks]

    image_ids = []

    for chunk in chunks:
        for image_id in chunk.get("image_ids", []):
            image_ids.append(int(image_id))

    if integrate_each_keyframe_once:
        # Preserve first-seen order while removing duplicates.
        seen = set()
        unique_ids = []
        for image_id in image_ids:
            if image_id in seen:
                continue
            seen.add(image_id)
            unique_ids.append(image_id)
        image_ids = unique_ids

    return image_ids


def _load_keyframe_rgbd_camera(
    keyframe_dir: Path,
    image_id: int,
    min_depth: float,
    depth_trunc: float,
    use_pose_w2c_if_available: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one saved keyframe.

    Returns:
        color:
            RGB uint8, shape HxWx3.

        depth:
            float32 depth, shape HxW.

        intrinsic:
            3x3 K matrix.

        extrinsic_w2c:
            4x4 world-to-camera matrix for Open3D TSDF integration.
    """

    prefix = keyframe_dir / f"kf_{image_id:06d}"

    rgb_path = Path(str(prefix) + "_rgb.png")
    depth_path = Path(str(prefix) + "_depth.npy")
    intrinsic_path = Path(str(prefix) + "_intrinsics.npy")
    pose_w2c_path = Path(str(prefix) + "_pose_w2c.npy")
    pose_c2w_path = Path(str(prefix) + "_pose_c2w.npy")

    if not rgb_path.exists():
        raise FileNotFoundError(f"missing RGB: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"missing depth: {depth_path}")
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"missing intrinsics: {intrinsic_path}")

    color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise RuntimeError(f"failed to read RGB: {rgb_path}")

    # Open3D color image should be RGB.
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    depth = np.load(str(depth_path)).astype(np.float32)
    intrinsic = np.load(str(intrinsic_path)).astype(np.float64)

    if color_rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"RGB/depth size mismatch for kf {image_id}: " f"rgb={color_rgb.shape[:2]}, depth={depth.shape[:2]}"
        )

    if intrinsic.shape != (3, 3):
        raise ValueError(f"Invalid intrinsic shape for kf {image_id}: {intrinsic.shape}")

    depth = np.asarray(depth, dtype=np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= min_depth] = 0.0
    depth[depth >= depth_trunc] = 0.0

    if use_pose_w2c_if_available and pose_w2c_path.exists():
        extrinsic_w2c = np.load(str(pose_w2c_path)).astype(np.float64)
    elif pose_c2w_path.exists():
        pose_c2w = np.load(str(pose_c2w_path)).astype(np.float64)
        extrinsic_w2c = np.linalg.inv(pose_c2w)
    else:
        raise FileNotFoundError(
            f"missing pose for kf {image_id}: expected " f"{pose_w2c_path.name} or {pose_c2w_path.name}"
        )

    if extrinsic_w2c.shape != (4, 4):
        raise ValueError(f"Invalid extrinsic shape for kf {image_id}: {extrinsic_w2c.shape}")

    return color_rgb, depth, intrinsic, extrinsic_w2c


# python da3_slam/open3d_tsdf.py
if __name__ == "__main__":
    mesh, pcd, stats = build_open3d_tsdf_from_saved_keyframes(
        map_dir="output_da3_map",
        output_mesh_path="output_da3_map/tsdf_latest_mesh.ply",
        output_pcd_path="output_da3_map/tsdf_latest_cloud.ply",
        use_chunks=None,  # all chunks
        voxel_length=0.02,
        sdf_trunc=0.10,
        depth_trunc=5.0,
    )
