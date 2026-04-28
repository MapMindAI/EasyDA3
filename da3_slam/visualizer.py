import threading
import time
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


class MatplotlibTrajectoryVisualizer:
    def __init__(
        self,
        mode: str = "3d",  # "3d" or "2d"
        plane: str = "xz",  # used only in 2d mode
        update_hz: float = 5.0,
        window_title: str = "DA3-SLAM Trajectory",
        draw_camera_direction: bool = True,
        auto_scale_margin: float = 0.5,
    ):
        """
        Realtime trajectory visualizer using matplotlib.

        Args:
            mode:
                "3d" or "2d"

            plane:
                For 2D mode:
                    "xz", "xy", or "yz"

            update_hz:
                Refresh rate.

            window_title:
                Figure title.

            draw_camera_direction:
                Draw latest camera forward direction.

            auto_scale_margin:
                Extra margin added to plot bounds.
        """
        if mode not in ("3d", "2d"):
            raise ValueError("mode must be '3d' or '2d'")

        if plane not in ("xz", "xy", "yz"):
            raise ValueError("plane must be 'xz', 'xy', or 'yz'")

        self.mode = mode
        self.plane = plane
        self.update_hz = float(update_hz)
        self.window_title = window_title
        self.draw_camera_direction = draw_camera_direction
        self.auto_scale_margin = float(auto_scale_margin)

        self._lock = threading.Lock()

        self._poses_c2w: Dict[int, np.ndarray] = {}
        self._latest_frame_id: Optional[int] = None
        self._latest_keyframe_id: Optional[int] = None
        self._latest_chunk_id: Optional[int] = None

        self._running = False
        self._thread = None

        self.fig = None
        self.ax = None

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.fig is not None:
            plt.close(self.fig)

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
        plt.ion()

        if self.mode == "3d":
            self.fig = plt.figure(self.window_title, figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.fig = plt.figure(self.window_title, figsize=(8, 8))
            self.ax = self.fig.add_subplot(111)

        self.fig.canvas.manager.set_window_title(self.window_title)

        period = 1.0 / max(self.update_hz, 1e-6)

        while self._running:
            self._draw_once()
            plt.pause(0.001)
            time.sleep(period)

        plt.close(self.fig)

    def _draw_once(self):
        with self._lock:
            poses = {k: v.copy() for k, v in self._poses_c2w.items()}
            latest_frame_id = self._latest_frame_id
            latest_keyframe_id = self._latest_keyframe_id
            latest_chunk_id = self._latest_chunk_id

        self.ax.cla()

        if len(poses) == 0:
            self.ax.set_title(
                f"{self.window_title}\n"
                f"Waiting for optimized poses...\n"
                f"frame={latest_frame_id}, keyframe={latest_keyframe_id}, chunk={latest_chunk_id}"
            )
            if self.mode == "3d":
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.set_zlabel("Z")
            else:
                self._set_2d_labels()
            return

        ids = sorted(poses.keys())

        traj = np.array([poses[i][:3, 3] for i in ids], dtype=np.float64)
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]

        if self.mode == "3d":
            self.ax.plot(x, y, z, linewidth=2, label="trajectory")
            self.ax.scatter(x, y, z, s=10)

            # current pose
            x0, y0, z0 = traj[-1]
            self.ax.scatter([x0], [y0], [z0], s=80, c="red", label="current")

            if self.draw_camera_direction:
                R = poses[ids[-1]][:3, :3]
                forward = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                self.ax.quiver(
                    [x0],
                    [y0],
                    [z0],
                    [forward[0]],
                    [forward[1]],
                    [forward[2]],
                    length=0.5,
                    normalize=True,
                )

            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.legend(loc="upper right")
            self._set_equal_3d_axes(x, y, z)

        else:
            u, v = self._select_2d_plane(x, y, z)
            self.ax.plot(u, v, linewidth=2)
            self.ax.scatter(u, v, s=10)

            u0, v0 = u[-1], v[-1]
            self.ax.scatter([u0], [v0], s=80, c="red")

            if self.draw_camera_direction:
                R = poses[ids[-1]][:3, :3]
                forward = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                p = traj[-1]
                p1 = p + 0.5 * forward

                u1, v1 = self._select_2d_plane(
                    np.array([p[0], p1[0]]),
                    np.array([p[1], p1[1]]),
                    np.array([p[2], p1[2]]),
                )
                self.ax.arrow(
                    u1[0],
                    v1[0],
                    u1[1] - u1[0],
                    v1[1] - v1[0],
                    head_width=0.05,
                    length_includes_head=True,
                )

            self._set_2d_labels()
            self.ax.axis("equal")
            self.ax.grid(True)

        self.ax.set_title(
            f"{self.window_title}\n"
            f"poses={len(poses)}, latest optimized kf={ids[-1]}, "
            f"frame={latest_frame_id}, keyframe={latest_keyframe_id}, chunk={latest_chunk_id}"
        )

    def _select_2d_plane(self, x, y, z):
        if self.plane == "xz":
            return x, z
        if self.plane == "xy":
            return x, y
        if self.plane == "yz":
            return y, z
        raise RuntimeError("Invalid plane")

    def _set_2d_labels(self):
        if self.plane == "xz":
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Z")
        elif self.plane == "xy":
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
        elif self.plane == "yz":
            self.ax.set_xlabel("Y")
            self.ax.set_ylabel("Z")

    def _set_equal_3d_axes(self, x, y, z):
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)

        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        z_mid = 0.5 * (z_min + z_max)

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        half = 0.5 * max(x_range, y_range, z_range, 1e-3) + self.auto_scale_margin

        self.ax.set_xlim(x_mid - half, x_mid + half)
        self.ax.set_ylim(y_mid - half, y_mid + half)
        self.ax.set_zlim(z_mid - half, z_mid + half)
