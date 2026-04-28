from __future__ import annotations

import threading
import time
from typing import Dict, Optional

import numpy as np
import open3d as o3d

try:
    from .logging_utils import get_logger
except ImportError:
    from logging_utils import get_logger


logger = get_logger(__name__)


class Open3DTrajectoryVisualizer:
    def __init__(
        self,
        update_hz: float = 10.0,
        window_title: str = "DA3-SLAM Trajectory",
        camera_size: float = 0.25,
        trajectory_color=(0.1, 0.55, 1.0),
        current_camera_color=(1.0, 0.15, 0.05),
        history_camera_color=(0.35, 0.35, 0.35),
        draw_history_camera_stride: int = 10,
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

        self._lock = threading.Lock()
        self._poses_c2w: Dict[int, np.ndarray] = {}
        self._latest_frame_id: Optional[int] = None
        self._latest_keyframe_id: Optional[int] = None
        self._latest_chunk_id: Optional[int] = None

        self._running = False
        self._thread = None

        self._vis = None
        self._trajectory = o3d.geometry.LineSet()
        self._current_camera = o3d.geometry.LineSet()
        self._history_cameras = o3d.geometry.LineSet()
        self._origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self._geometry_added = False

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
        self.update_snapshot(
            poses_c2w=snapshot["poses_c2w"],
            latest_frame_id=snapshot["frame_id"],
            latest_keyframe_id=snapshot["latest_keyframe_id"],
            latest_chunk_id=snapshot["latest_chunk_id"],
        )

    def update_snapshot(
        self,
        poses_c2w: Dict[int, np.ndarray],
        latest_frame_id: Optional[int] = None,
        latest_keyframe_id: Optional[int] = None,
        latest_chunk_id: Optional[int] = None,
    ):
        with self._lock:
            self._poses_c2w = {int(k): np.asarray(v, dtype=np.float64).copy() for k, v in poses_c2w.items()}
            self._latest_frame_id = latest_frame_id
            self._latest_keyframe_id = latest_keyframe_id
            self._latest_chunk_id = latest_chunk_id

    def _run_loop(self):
        self._vis = o3d.visualization.Visualizer()
        created = self._vis.create_window(window_name=self.window_title, width=1280, height=800)
        if not created:
            logger.error("Failed to create Open3D visualizer window")
            self._running = False
            return

        try:
            self._setup_scene()
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
        render_option.background_color = np.array([0.02, 0.02, 0.025], dtype=np.float64)
        render_option.line_width = 3.0
        render_option.point_size = 5.0

        self._vis.add_geometry(self._origin)
        self._vis.add_geometry(self._trajectory)
        self._vis.add_geometry(self._history_cameras)
        self._vis.add_geometry(self._current_camera)
        self._geometry_added = True

    def _draw_once(self):
        with self._lock:
            poses = {k: v.copy() for k, v in self._poses_c2w.items()}
            latest_frame_id = self._latest_frame_id
            latest_keyframe_id = self._latest_keyframe_id
            latest_chunk_id = self._latest_chunk_id

        if not poses:
            return

        ids = sorted(poses.keys())
        trajectory_points = np.array([poses[i][:3, 3] for i in ids], dtype=np.float64)

        self._update_trajectory(trajectory_points)
        self._update_current_camera(poses[ids[-1]])
        self._update_history_cameras([poses[i] for i in ids])
        self._update_window_title(
            pose_count=len(ids),
            latest_pose_id=ids[-1],
            latest_frame_id=latest_frame_id,
            latest_keyframe_id=latest_keyframe_id,
            latest_chunk_id=latest_chunk_id,
        )

        self._vis.update_geometry(self._trajectory)
        self._vis.update_geometry(self._history_cameras)
        self._vis.update_geometry(self._current_camera)

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


# Compatibility for old imports. The implementation is now Open3D-based.
MatplotlibTrajectoryVisualizer = Open3DTrajectoryVisualizer
