from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import open3d as o3d

try:
    from .logging_utils import get_logger
    from .open3d_tsdf import _load_keyframe_rgbd_camera
except ImportError:
    from logging_utils import get_logger
    from open3d_tsdf import _load_keyframe_rgbd_camera


logger = get_logger(__name__)


class Open3DTrajectoryVisualizer:
    def __init__(
        self,
        update_hz: float = 10.0,
        window_title: str = "DA3-SLAM Trajectory",
        camera_size: float = 0.25,
        trajectory_color=(0.1, 0.55, 1.0),
        current_camera_color=(0.0, 0.0, 0.0),
        history_camera_color=(0.0, 0.0, 0.0),
        draw_history_camera_stride: int = 10,
        max_render_z_far: float = 10000.0,
        map_dir: Optional[str] = None,
        live_chunk_tsdf: bool = True,
        tsdf_update_hz: float = 1.0,
        tsdf_voxel_length: float = 0.03,
        tsdf_sdf_trunc: float = 0.12,
        tsdf_depth_trunc: float = 200.0,
        tsdf_depth_scale: float = 1.0,
        tsdf_volume_unit_resolution: int = 16,
        tsdf_depth_sampling_stride: int = 1,
        tsdf_min_depth: float = 1e-4,
        tsdf_min_confidence: float = 2.0,
        tsdf_use_pose_w2c_if_available: bool = True,
        max_live_chunks: int = 20,
        follow_smoothing_alpha: float = 0.18,
    ):
        """
        Realtime Open3D trajectory visualizer.

        Draws:
            1. Optimized trajectory as a line strip.
            2. Current optimized camera pose as a red camera frustum.
            3. Optional sparse history camera frustums.
        """

        self.update_hz = float(update_hz)
        self.window_title = window_title
        self.camera_size = float(camera_size)
        self.trajectory_color = np.asarray(trajectory_color, dtype=np.float64)
        self.current_camera_color = np.asarray(current_camera_color, dtype=np.float64)
        self.history_camera_color = np.asarray(history_camera_color, dtype=np.float64)
        self.draw_history_camera_stride = int(max(0, draw_history_camera_stride))
        self.max_render_z_far = float(max_render_z_far)
        self.live_chunk_tsdf = bool(live_chunk_tsdf)
        self.tsdf_update_period = 1.0 / max(1e-6, float(tsdf_update_hz))
        self.tsdf_voxel_length = float(tsdf_voxel_length)
        self.tsdf_sdf_trunc = float(tsdf_sdf_trunc)
        self.tsdf_depth_trunc = float(tsdf_depth_trunc)
        self.tsdf_depth_scale = float(tsdf_depth_scale)
        self.tsdf_volume_unit_resolution = int(tsdf_volume_unit_resolution)
        self.tsdf_depth_sampling_stride = int(tsdf_depth_sampling_stride)
        self.tsdf_min_depth = float(tsdf_min_depth)
        self.tsdf_min_confidence = float(tsdf_min_confidence)
        self.tsdf_use_pose_w2c_if_available = bool(tsdf_use_pose_w2c_if_available)
        self.map_dir = None if map_dir is None else Path(map_dir)
        self.max_live_chunks = int(max(1, max_live_chunks))
        self.follow_smoothing_alpha = float(np.clip(follow_smoothing_alpha, 0.01, 1.0))

        self._lock = threading.Lock()
        self._poses_c2w: Dict[int, np.ndarray] = {}
        self._current_pose_c2w: Optional[np.ndarray] = None
        self._latest_frame_id: Optional[int] = None
        self._latest_keyframe_id: Optional[int] = None
        self._latest_chunk_id: Optional[int] = None

        self._running = False
        self._thread = None

        self._vis = None
        self._trajectory = o3d.geometry.LineSet()
        self._current_camera = o3d.geometry.LineSet()
        self._history_cameras = o3d.geometry.LineSet()
        self._origin = self._make_camera_coordinate_origin_axes(axis_size=0.5)
        self._geometry_added = False
        self._chunk_meshes: Dict[int, o3d.geometry.TriangleMesh] = {}
        self._chunk_pointclouds: Dict[int, o3d.geometry.PointCloud] = {}
        self._chunk_signatures: Dict[int, Tuple] = {}
        self._next_tsdf_sync_time = 0.0
        self._follow_latest_pose = True
        self._show_chunk_mesh = True
        self._show_chunk_pointcloud = False
        self._follow_reference_pose_c2w: Optional[np.ndarray] = None
        self._follow_reference_lookat: Optional[np.ndarray] = None
        self._follow_smoothed_lookat: Optional[np.ndarray] = None

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="Open3DTrajectoryVisualizer", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def update_from_pipeline(self, pipeline):
        snapshot = pipeline.get_trajectory_snapshot()
        if self.map_dir is None and hasattr(pipeline, "output_dir"):
            self.map_dir = Path(getattr(pipeline, "output_dir"))
        self.update_snapshot(
            poses_c2w=snapshot["poses_c2w"],
            current_pose_c2w=snapshot.get("current_pose_c2w"),
            latest_frame_id=snapshot["frame_id"],
            latest_keyframe_id=snapshot["latest_keyframe_id"],
            latest_chunk_id=snapshot["latest_chunk_id"],
        )

    def update_snapshot(
        self,
        poses_c2w: Dict[int, np.ndarray],
        current_pose_c2w: Optional[np.ndarray] = None,
        latest_frame_id: Optional[int] = None,
        latest_keyframe_id: Optional[int] = None,
        latest_chunk_id: Optional[int] = None,
    ):
        with self._lock:
            self._poses_c2w = {int(k): np.asarray(v, dtype=np.float64).copy() for k, v in poses_c2w.items()}
            self._current_pose_c2w = (
                None if current_pose_c2w is None else np.asarray(current_pose_c2w, dtype=np.float64).copy()
            )
            self._latest_frame_id = latest_frame_id
            self._latest_keyframe_id = latest_keyframe_id
            self._latest_chunk_id = latest_chunk_id

    def _run_loop(self):
        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        created = self._vis.create_window(window_name=self.window_title, width=1280, height=800)
        if not created:
            logger.error("Failed to create Open3D visualizer window")
            self._running = False
            return

        try:
            self._setup_scene()
            self._register_key_callbacks()
            period = 1.0 / max(self.update_hz, 1e-6)

            while self._running:
                self._draw_once()
                if not self._vis.poll_events():
                    self._running = False
                    break
                self._vis.update_renderer()
                time.sleep(period)
        finally:
            self._vis.destroy_window()
            self._vis = None
            self._geometry_added = False

    def _setup_scene(self):
        render_option = self._vis.get_render_option()
        render_option.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        render_option.line_width = 3.0
        render_option.point_size = 5.0
        view_control = self._vis.get_view_control()
        if view_control is not None:
            try:
                view_control.set_constant_z_far(self.max_render_z_far)
            except Exception:
                logger.debug("Open3D ViewControl does not support set_constant_z_far on this build.")

        self._vis.add_geometry(self._origin)
        self._vis.add_geometry(self._trajectory)
        self._vis.add_geometry(self._history_cameras)
        self._vis.add_geometry(self._current_camera)
        self._geometry_added = True
        ui_text = (
            "\n"
            "========== Open3D Visualizer Controls ==========\n"
            "  F : toggle follow latest pose\n"
            "  M : toggle chunk TSDF mesh\n"
            "  P : toggle chunk TSDF point cloud\n"
            "================================================\n"
        )
        print(ui_text, flush=True)
        logger.info("Open3D controls: [F]=follow latest pose, [M]=toggle chunk mesh, [P]=toggle chunk point cloud")

    def _register_key_callbacks(self):
        self._vis.register_key_callback(ord("F"), self._on_toggle_follow_pose)
        self._vis.register_key_callback(ord("M"), self._on_toggle_chunk_mesh)
        self._vis.register_key_callback(ord("P"), self._on_toggle_chunk_pointcloud)

    def _on_toggle_follow_pose(self, vis):
        self._follow_latest_pose = not self._follow_latest_pose
        if self._follow_latest_pose:
            self._reset_follow_anchor()
        logger.info("Follow latest pose: %s", self._follow_latest_pose)
        return False

    def _on_toggle_chunk_mesh(self, vis):
        self._show_chunk_mesh = not self._show_chunk_mesh
        logger.info("Show chunk mesh: %s", self._show_chunk_mesh)
        for chunk_id, mesh in list(self._chunk_meshes.items()):
            if self._show_chunk_mesh:
                vis.add_geometry(mesh, reset_bounding_box=False)
            else:
                vis.remove_geometry(mesh, reset_bounding_box=False)
        return False

    def _on_toggle_chunk_pointcloud(self, vis):
        self._show_chunk_pointcloud = not self._show_chunk_pointcloud
        logger.info("Show chunk point cloud: %s", self._show_chunk_pointcloud)
        for chunk_id, pcd in list(self._chunk_pointclouds.items()):
            if self._show_chunk_pointcloud:
                vis.add_geometry(pcd, reset_bounding_box=False)
            else:
                vis.remove_geometry(pcd, reset_bounding_box=False)
        return False

    def _draw_once(self):
        with self._lock:
            poses = {k: v.copy() for k, v in self._poses_c2w.items()}
            current_pose_c2w = None if self._current_pose_c2w is None else self._current_pose_c2w.copy()
            latest_frame_id = self._latest_frame_id
            latest_keyframe_id = self._latest_keyframe_id
            latest_chunk_id = self._latest_chunk_id

        if not poses:
            return

        ids = sorted(poses.keys())
        trajectory_points = np.array([poses[i][:3, 3] for i in ids], dtype=np.float64)

        self._update_trajectory(trajectory_points)
        pose_for_current_camera = poses[ids[-1]] if current_pose_c2w is None else current_pose_c2w
        self._update_current_camera(pose_for_current_camera)
        if self._follow_latest_pose:
            self._follow_pose(pose_for_current_camera)
        self._update_history_cameras([poses[i] for i in ids])
        self._update_window_title(
            pose_count=len(ids),
            latest_pose_id=ids[-1],
            latest_frame_id=latest_frame_id,
            latest_keyframe_id=latest_keyframe_id,
            latest_chunk_id=latest_chunk_id,
        )
        self._update_live_chunk_tsdf_meshes()

        self._vis.update_geometry(self._trajectory)
        self._vis.update_geometry(self._history_cameras)
        self._vis.update_geometry(self._current_camera)
        if self._show_chunk_mesh:
            for mesh in self._chunk_meshes.values():
                self._vis.update_geometry(mesh)
        if self._show_chunk_pointcloud:
            for pcd in self._chunk_pointclouds.values():
                self._vis.update_geometry(pcd)

    def _follow_pose(self, pose_c2w: np.ndarray):
        if self._vis is None:
            return

        vc = self._vis.get_view_control()
        if vc is None:
            return

        pose_c2w = np.asarray(pose_c2w, dtype=np.float64)
        if self._follow_reference_pose_c2w is None:
            self._init_follow_anchor(vc, pose_c2w)

        current_t = np.asarray(pose_c2w[:3, 3], dtype=np.float64)
        reference_t = np.asarray(self._follow_reference_pose_c2w[:3, 3], dtype=np.float64)
        delta_t = current_t - reference_t

        target_lookat = self._follow_reference_lookat + delta_t

        alpha = self.follow_smoothing_alpha
        if self._follow_smoothed_lookat is None:
            self._follow_smoothed_lookat = target_lookat.copy()
        else:
            self._follow_smoothed_lookat = (1.0 - alpha) * self._follow_smoothed_lookat + alpha * target_lookat

        try:
            vc.set_lookat(self._follow_smoothed_lookat)
        except Exception:
            pass

    def _reset_follow_anchor(self):
        self._follow_reference_pose_c2w = None
        self._follow_reference_lookat = None
        self._follow_smoothed_lookat = None

    def _init_follow_anchor(self, vc, pose_c2w: np.ndarray):
        self._follow_reference_pose_c2w = np.asarray(pose_c2w, dtype=np.float64).copy()

        lookat = None

        if hasattr(vc, "get_lookat"):
            try:
                lookat = np.asarray(vc.get_lookat(), dtype=np.float64).reshape(3)
            except Exception:
                lookat = None

        if lookat is None:
            lookat = np.asarray(pose_c2w[:3, 3], dtype=np.float64)

        self._follow_reference_lookat = lookat
        self._follow_smoothed_lookat = lookat.copy()

    def _update_trajectory(self, points: np.ndarray):
        if len(points) == 0:
            self._set_lineset(self._trajectory, np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), [])
            return

        lines = np.array([[idx, idx + 1] for idx in range(len(points) - 1)], dtype=np.int32)
        colors = np.tile(self.trajectory_color.reshape(1, 3), (len(lines), 1))
        self._set_lineset(self._trajectory, points, lines, colors)

    def _update_current_camera(self, pose_c2w: np.ndarray):
        points, lines = self._make_camera_frustum_lines(pose_c2w, scale=self.camera_size, include_axes=True)
        colors = np.tile(self.current_camera_color.reshape(1, 3), (len(lines), 1))
        if len(lines) >= 12:
            colors[-3:] = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.1, 0.35, 1.0],
                ],
                dtype=np.float64,
            )
        self._set_lineset(self._current_camera, points, lines, colors)

    def _update_history_cameras(self, poses_c2w):
        if self.draw_history_camera_stride <= 0:
            self._set_lineset(self._history_cameras, np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), [])
            return

        selected = poses_c2w[: -1 : self.draw_history_camera_stride]
        if not selected:
            self._set_lineset(self._history_cameras, np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), [])
            return

        all_points = []
        all_lines = []
        offset = 0
        for pose in selected:
            points, lines = self._make_camera_frustum_lines(pose, scale=self.camera_size * 0.55)
            all_points.append(points)
            all_lines.append(lines + offset)
            offset += len(points)

        points = np.vstack(all_points)
        lines = np.vstack(all_lines)
        colors = np.tile(self.history_camera_color.reshape(1, 3), (len(lines), 1))
        self._set_lineset(self._history_cameras, points, lines, colors)

    def _update_window_title(
        self,
        pose_count: int,
        latest_pose_id: int,
        latest_frame_id: Optional[int],
        latest_keyframe_id: Optional[int],
        latest_chunk_id: Optional[int],
    ):
        title = (
            f"{self.window_title} | poses={pose_count}, latest optimized kf={latest_pose_id}, "
            f"frame={latest_frame_id}, keyframe={latest_keyframe_id}, chunk={latest_chunk_id}"
        )
        # Open3D's legacy Visualizer has no portable runtime title setter.
        # Keep metadata in logs every time a new optimized pose arrives.
        if not hasattr(self, "_last_logged_pose_id") or self._last_logged_pose_id != latest_pose_id:
            logger.info(title)
            self._last_logged_pose_id = latest_pose_id

    @staticmethod
    def _make_camera_frustum_lines(pose_c2w: np.ndarray, scale: float, include_axes: bool = False):
        pose_c2w = np.asarray(pose_c2w, dtype=np.float64)

        z = float(scale)
        x = z * 0.7
        y = z * 0.45

        local_points = [
            [0.0, 0.0, 0.0],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
            [0.0, 0.0, z * 1.35],
        ]

        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
            [0, 5],
        ]

        if include_axes:
            axis_len = z * 1.2
            local_points.extend(
                [
                    [axis_len, 0.0, 0.0],
                    [0.0, axis_len, 0.0],
                    [0.0, 0.0, axis_len],
                ]
            )
            lines.extend([[0, 6], [0, 7], [0, 8]])

        local_points = np.array(local_points, dtype=np.float64)
        lines = np.array(lines, dtype=np.int32)

        local_points_h = np.hstack([local_points, np.ones((len(local_points), 1), dtype=np.float64)])
        world_points = (pose_c2w @ local_points_h.T).T[:, :3]

        return world_points, lines

    @staticmethod
    def _set_lineset(lineset: o3d.geometry.LineSet, points, lines, colors):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        lines = np.asarray(lines, dtype=np.int32).reshape(-1, 2)
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)

        lineset.points = o3d.utility.Vector3dVector(points)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector(colors)

    def _update_live_chunk_tsdf_meshes(self):
        if not self.live_chunk_tsdf or self.map_dir is None:
            return
        if self._vis is None:
            return

        now = time.time()
        if now < self._next_tsdf_sync_time:
            return
        self._next_tsdf_sync_time = now + self.tsdf_update_period

        manifest_path = self.map_dir / "manifest.json"
        keyframe_dir = self.map_dir / "keyframes"
        if not manifest_path.exists() or not keyframe_dir.exists():
            return

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read manifest for live TSDF update: %s", exc)
            return

        chunks = manifest.get("chunks", [])
        if not isinstance(chunks, list):
            return
        chunks = sorted(chunks, key=lambda c: int(c.get("chunk_id", -1)))
        if len(chunks) > self.max_live_chunks:
            chunks = chunks[-self.max_live_chunks :]

        active_chunk_ids = set()
        for chunk in chunks:
            chunk_id = int(chunk.get("chunk_id", -1))
            if chunk_id < 0:
                continue
            image_ids = [int(x) for x in chunk.get("image_ids", [])]
            if len(image_ids) == 0:
                continue

            signature = self._make_chunk_signature(image_ids, keyframe_dir)
            active_chunk_ids.add(chunk_id)

            if self._chunk_signatures.get(chunk_id) == signature:
                continue

            mesh, pcd = self._build_chunk_tsdf_geometry(image_ids, keyframe_dir)
            if mesh is None and pcd is None:
                continue

            old_mesh = self._chunk_meshes.get(chunk_id)
            if old_mesh is not None:
                self._vis.remove_geometry(old_mesh, reset_bounding_box=False)
            old_pcd = self._chunk_pointclouds.get(chunk_id)
            if old_pcd is not None:
                self._vis.remove_geometry(old_pcd, reset_bounding_box=False)

            if mesh is not None:
                self._chunk_meshes[chunk_id] = mesh
            else:
                self._chunk_meshes.pop(chunk_id, None)
            if pcd is not None:
                self._chunk_pointclouds[chunk_id] = pcd
            else:
                self._chunk_pointclouds.pop(chunk_id, None)
            self._chunk_signatures[chunk_id] = signature
            if mesh is not None and self._show_chunk_mesh:
                self._vis.add_geometry(mesh, reset_bounding_box=False)
            if pcd is not None and self._show_chunk_pointcloud:
                self._vis.add_geometry(pcd, reset_bounding_box=False)
            logger.info("Updated TSDF mesh for chunk %s (%s keyframes).", chunk_id, len(image_ids))

        stale_ids = [chunk_id for chunk_id in self._chunk_meshes.keys() if chunk_id not in active_chunk_ids]
        for chunk_id in stale_ids:
            mesh = self._chunk_meshes.pop(chunk_id)
            self._vis.remove_geometry(mesh, reset_bounding_box=False)
        stale_pcd_ids = [chunk_id for chunk_id in self._chunk_pointclouds.keys() if chunk_id not in active_chunk_ids]
        for chunk_id in stale_pcd_ids:
            pcd = self._chunk_pointclouds.pop(chunk_id)
            self._vis.remove_geometry(pcd, reset_bounding_box=False)
        stale_signature_ids = [chunk_id for chunk_id in self._chunk_signatures.keys() if chunk_id not in active_chunk_ids]
        for chunk_id in stale_signature_ids:
            self._chunk_signatures.pop(chunk_id, None)

    def _make_chunk_signature(self, image_ids: Sequence[int], keyframe_dir: Path) -> Tuple:
        signature: List[Tuple[int, float, float, float, float, float]] = []
        for image_id in image_ids:
            prefix = keyframe_dir / f"kf_{int(image_id):06d}"
            depth_mtime = self._safe_mtime(Path(str(prefix) + "_depth.npy"))
            depth_conf_mtime = self._safe_mtime(Path(str(prefix) + "_depth_conf.npy"))
            intrinsics_mtime = self._safe_mtime(Path(str(prefix) + "_intrinsics.npy"))
            pose_w2c_mtime = self._safe_mtime(Path(str(prefix) + "_pose_w2c.npy"))
            pose_c2w_mtime = self._safe_mtime(Path(str(prefix) + "_pose_c2w.npy"))
            signature.append(
                (int(image_id), depth_mtime, depth_conf_mtime, intrinsics_mtime, pose_w2c_mtime, pose_c2w_mtime)
            )
        return tuple(signature)

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except FileNotFoundError:
            return -1.0

    def _build_chunk_tsdf_geometry(
        self,
        image_ids: Sequence[int],
        keyframe_dir: Path,
    ) -> Tuple[Optional[o3d.geometry.TriangleMesh], Optional[o3d.geometry.PointCloud]]:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.tsdf_voxel_length,
            sdf_trunc=self.tsdf_sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            volume_unit_resolution=self.tsdf_volume_unit_resolution,
            depth_sampling_stride=self.tsdf_depth_sampling_stride,
        )

        integrated = 0
        for image_id in image_ids:
            try:
                color_rgb, depth, intrinsic, extrinsic_w2c = _load_keyframe_rgbd_camera(
                    keyframe_dir=keyframe_dir,
                    image_id=int(image_id),
                    min_depth=self.tsdf_min_depth,
                    min_confidence=self.tsdf_min_confidence,
                    depth_trunc=self.tsdf_depth_trunc,
                    use_pose_w2c_if_available=self.tsdf_use_pose_w2c_if_available,
                )
            except Exception:
                continue

            h, w = depth.shape[:2]
            o3d_color = o3d.geometry.Image(color_rgb)
            o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=o3d_color,
                depth=o3d_depth,
                depth_scale=self.tsdf_depth_scale,
                depth_trunc=self.tsdf_depth_trunc,
                convert_rgb_to_intensity=False,
            )

            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=int(w),
                height=int(h),
                fx=float(intrinsic[0, 0]),
                fy=float(intrinsic[1, 1]),
                cx=float(intrinsic[0, 2]),
                cy=float(intrinsic[1, 2]),
            )
            volume.integrate(rgbd, o3d_intrinsic, extrinsic_w2c.astype(np.float64))
            integrated += 1

        if integrated == 0:
            return None, None

        mesh = volume.extract_triangle_mesh()
        if len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
        else:
            mesh = None

        pcd = volume.extract_point_cloud()
        if len(pcd.points) == 0:
            pcd = None

        return mesh, pcd

    @staticmethod
    def _make_camera_coordinate_origin_axes(axis_size: float) -> o3d.geometry.LineSet:
        """
        Camera-coordinate origin axes:
            X: right
            Y: down
            Z: front
        """
        s = float(axis_size)
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # origin
                [s, 0.0, 0.0],  # +X (right)
                [0.0, -s, 0.0],  # +Y (down) with Open3D's +Y-up world
                [0.0, 0.0, s],  # +Z (front)
            ],
            dtype=np.float64,
        )
        lines = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
            ],
            dtype=np.int32,
        )
        colors = np.array(
            [
                [1.0, 0.0, 0.0],  # X red
                [0.0, 1.0, 0.0],  # Y green
                [0.1, 0.35, 1.0],  # Z blue
            ],
            dtype=np.float64,
        )

        axes = o3d.geometry.LineSet()
        axes.points = o3d.utility.Vector3dVector(points)
        axes.lines = o3d.utility.Vector2iVector(lines)
        axes.colors = o3d.utility.Vector3dVector(colors)
        return axes


# Compatibility for old imports. The implementation is now Open3D-based.
MatplotlibTrajectoryVisualizer = Open3DTrajectoryVisualizer
