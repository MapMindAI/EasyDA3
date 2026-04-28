from __future__ import annotations

import json
import queue
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from typing import Any, Dict, Set, List, Optional, Sequence

import cv2
import numpy as np
import threading

from optical_frontend import (
    OpticalFlowKeyframeProcessor,
    OpticalFlowResult,
    visualize_optical_flow_result,
)
from da3_client import DepthAnything3
from backend.da3_pose_graph_optimizer import (
    DA3ChunkPoseGraphOptimizer,
    PoseChunk,
    ImagePosePrior,
)
from visualizer import MatplotlibTrajectoryVisualizer


@dataclass
class KeyframeRecord:
    image_id: int
    source_frame_id: int
    image_bgr: np.ndarray

    depth: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None

    # Optimized global camera pose.
    pose_c2w: Optional[np.ndarray] = None
    pose_w2c: Optional[np.ndarray] = None

    # Latest DA3 local result for this keyframe.
    da3_local_w2c: Optional[np.ndarray] = None
    last_chunk_id: Optional[int] = None


@dataclass
class DA3ChunkRecord:
    chunk_id: int
    image_ids: List[int]

    # DA3 outputs.
    depth_list: List[np.ndarray]
    intrinsics_list: List[np.ndarray]
    extrinsics_list: List[np.ndarray]

    # Current estimated chunk scale.
    scale: float = 1.0

    # Optimization bookkeeping.
    initial_error: Optional[float] = None
    final_error: Optional[float] = None


@dataclass
class DA3BackendJob:
    chunk_id: int
    image_ids: List[int]
    images_bgr: List[np.ndarray]


@dataclass
class DA3BackendResult:
    chunk_id: int
    image_ids: List[int]
    optimized_image_ids: List[int] = field(default_factory=list)
    error: Optional[str] = None
    traceback_text: Optional[str] = None


@dataclass
class MappingProcessResult:
    frame_id: int
    is_keyframe: bool
    keyframe_id: Optional[int]

    da3_ran: bool = False
    new_chunk_id: Optional[int] = None
    completed_chunk_ids: List[int] = field(default_factory=list)

    da3_scheduled: bool = False
    scheduled_chunk_id: Optional[int] = None

    optimization_ran: bool = False
    optimized_image_ids: List[int] = field(default_factory=list)

    backend_error: Optional[str] = None

    flow_result: Any = None


class DA3StreamingMappingPipeline:
    def __init__(
        self,
        flow_processor,
        da3_client,
        pose_graph_optimizer,
        output_dir: str | Path,
        window_size: int = 10,
        da3_stride_new_keyframes: int = 3,
        optimizer_num_chunks: int = 5,
        anchor_prior_weight: float = 1e6,
        save_depth_png_preview: bool = True,
    ):
        """
        Streaming visual mapping pipeline.

        Args:
            flow_processor:
                OpticalFlowKeyframeProcessor instance.

            da3_client:
                DepthAnything3 client instance.
                It must provide:
                    da3_client.run(list_of_images)

            pose_graph_optimizer:
                DA3ChunkPoseGraphOptimizer instance.

                Important:
                    If DA3 extrinsics are world-to-camera, initialize optimizer with:
                        DA3ChunkPoseGraphOptimizer(input_pose_is_w2c=True)

            output_dir:
                Directory where keyframes, depth, intrinsics and poses are saved.

            window_size:
                N. Number of keyframes passed into each DA3 call.
                Default: 10.

            da3_stride_new_keyframes:
                M. Run DA3 after this many new keyframes since last DA3 call.
                Default: 3.

            optimizer_num_chunks:
                K. Optimize with the latest K DA3 chunks.
                Default: 5.

            anchor_prior_weight:
                Strong pose-prior weight used to fix the first chunk in the
                latest K chunks.

            save_depth_png_preview:
                If True, also saves an 8-bit visualization of depth.
        """

        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if da3_stride_new_keyframes < 1:
            raise ValueError("da3_stride_new_keyframes must be >= 1")
        if optimizer_num_chunks < 1:
            raise ValueError("optimizer_num_chunks must be >= 1")

        self.state_lock = threading.Lock()
        self.flow_processor = flow_processor
        self.da3_client = da3_client
        self.optimizer = pose_graph_optimizer

        self.window_size = int(window_size)
        self.da3_stride_new_keyframes = int(da3_stride_new_keyframes)
        self.optimizer_num_chunks = int(optimizer_num_chunks)
        self.anchor_prior_weight = float(anchor_prior_weight)
        self.scale_be_one_prior_weight = float(1e1)
        self.save_depth_png_preview = save_depth_png_preview
        self.min_valid_chunk_scale = 1e-4
        self.max_valid_chunk_scale = 1e4

        self.output_dir = Path(output_dir)
        self.keyframe_dir = self.output_dir / "keyframes"
        self.chunk_dir = self.output_dir / "chunks"

        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

        self.frame_id = -1
        self.next_keyframe_id = 0
        self.next_chunk_id = 0

        self.sliding_keyframe_ids = deque(maxlen=self.window_size)
        self.keyframes: Dict[int, KeyframeRecord] = {}
        self.chunks: List[DA3ChunkRecord] = []

        self.new_keyframes_since_last_da3 = 0

        # Global optimized results.
        self.optimized_pose_c2w: Dict[int, np.ndarray] = {}
        self.optimized_chunk_scales: Dict[int, float] = {}
        self.updated_keyframes: Set[int] = set()

        self.backend_job_queue: "queue.Queue[DA3BackendJob]" = queue.Queue(maxsize=1)
        self.backend_result_queue: "queue.Queue[DA3BackendResult]" = queue.Queue()
        self.backend_stop_event = threading.Event()
        self.backend_thread = threading.Thread(
            target=self._backend_worker_loop,
            name="DA3BackendWorker",
            daemon=True,
        )
        self.backend_thread.start()

    def _sanitize_chunk_scale(self, scale: float, default: float = 1.0) -> float:
        scale = float(scale)

        if not np.isfinite(scale):
            return default

        if scale <= self.min_valid_chunk_scale:
            return default

        if scale >= self.max_valid_chunk_scale:
            return default

        return scale

    def process_image(self, image: np.ndarray) -> MappingProcessResult:
        """
        Process one incoming image.

        Steps:
            1. Run optical-flow frontend.
            2. If keyframe, add to sliding window.
            3. If window is full and enough new keyframes exist, enqueue DA3.
            4. DA3, optimization, and backend saves run on the worker thread.
            5. Return any completed backend results from previous async jobs.

        Returns:
            MappingProcessResult
        """

        with self.state_lock:
            self.frame_id += 1
            frame_id = self.frame_id

        flow_result = self.flow_processor.process(image)

        output = MappingProcessResult(
            frame_id=frame_id,
            is_keyframe=bool(flow_result.is_keyframe),
            keyframe_id=None,
            flow_result=flow_result,
        )

        self._collect_backend_results(output)

        if not flow_result.is_keyframe:
            return output

        keyframe_id = self._add_keyframe(image, source_frame_id=frame_id)
        output.keyframe_id = keyframe_id

        if self._should_run_da3():
            scheduled_chunk_id = self._schedule_da3_on_current_window()
            if scheduled_chunk_id is not None:
                output.da3_scheduled = True
                output.scheduled_chunk_id = scheduled_chunk_id

        return output

    def close(self, wait: bool = False, drain: bool = False):
        if drain:
            self.backend_job_queue.join()

        self.backend_stop_event.set()

        if wait and self.backend_thread.is_alive():
            self.backend_thread.join()

    # ---------------------------------------------------------------------
    # Keyframe handling
    # ---------------------------------------------------------------------

    def _add_keyframe(self, image: np.ndarray, source_frame_id: int) -> int:
        image_bgr = self._to_bgr_uint8(image)

        with self.state_lock:
            image_id = self.next_keyframe_id
            self.next_keyframe_id += 1

            record = KeyframeRecord(
                image_id=image_id,
                source_frame_id=source_frame_id,
                image_bgr=image_bgr.copy(),
            )

            self.keyframes[image_id] = record
            self.sliding_keyframe_ids.append(image_id)
            self.new_keyframes_since_last_da3 += 1

        self._save_keyframe_rgb(record)

        return image_id

    def _should_run_da3(self) -> bool:
        with self.state_lock:
            if len(self.sliding_keyframe_ids) < self.window_size:
                return False

            if self.new_keyframes_since_last_da3 < self.da3_stride_new_keyframes:
                return False

            return True

    # ---------------------------------------------------------------------
    # DA3 chunk processing
    # ---------------------------------------------------------------------

    def _schedule_da3_on_current_window(self) -> Optional[int]:
        if self.backend_job_queue.full():
            return None

        with self.state_lock:
            if len(self.sliding_keyframe_ids) < self.window_size:
                return None

            if self.new_keyframes_since_last_da3 < self.da3_stride_new_keyframes:
                return None

            chunk_id = self.next_chunk_id
            image_ids = list(self.sliding_keyframe_ids)
            images = [self.keyframes[i].image_bgr.copy() for i in image_ids]
            previous_new_keyframes = self.new_keyframes_since_last_da3

            self.next_chunk_id += 1
            self.new_keyframes_since_last_da3 = 0

        job = DA3BackendJob(
            chunk_id=chunk_id,
            image_ids=image_ids,
            images_bgr=images,
        )

        try:
            self.backend_job_queue.put_nowait(job)
        except queue.Full:
            with self.state_lock:
                self.next_chunk_id -= 1
                self.new_keyframes_since_last_da3 = previous_new_keyframes
            return None

        return chunk_id

    def _run_da3_on_current_window(self) -> DA3ChunkRecord:
        with self.state_lock:
            chunk_id = self.next_chunk_id
            image_ids = list(self.sliding_keyframe_ids)
            images = [self.keyframes[i].image_bgr.copy() for i in image_ids]
            self.next_chunk_id += 1
            self.new_keyframes_since_last_da3 = 0

        job = DA3BackendJob(
            chunk_id=chunk_id,
            image_ids=image_ids,
            images_bgr=images,
        )

        return self._run_da3_job(job)

    def _run_da3_job(self, job: DA3BackendJob) -> DA3ChunkRecord:
        da3_result = self.da3_client.run(job.images_bgr)

        depth_list = [np.asarray(d, dtype=np.float32) for d in da3_result["depth_list"]]
        intrinsics_list = [np.asarray(k, dtype=np.float32) for k in da3_result["intrinsics_list"]]
        extrinsics_list = [self._as_4x4(np.asarray(e, dtype=np.float32)) for e in da3_result["extrinsics_list"]]

        if not (len(depth_list) == len(intrinsics_list) == len(extrinsics_list) == len(job.image_ids)):
            raise RuntimeError("DA3 output count does not match input keyframe count.")

        chunk = DA3ChunkRecord(
            chunk_id=job.chunk_id,
            image_ids=job.image_ids,
            depth_list=depth_list,
            intrinsics_list=intrinsics_list,
            extrinsics_list=extrinsics_list,
            scale=1.0,
        )

        with self.state_lock:
            self.chunks.append(chunk)

            for local_idx, image_id in enumerate(job.image_ids):
                kf = self.keyframes.get(image_id)
                if kf is None:
                    continue

                kf.depth = depth_list[local_idx]
                kf.intrinsics = intrinsics_list[local_idx]
                kf.da3_local_w2c = extrinsics_list[local_idx]
                kf.last_chunk_id = job.chunk_id
                self.updated_keyframes.add(image_id)

        self._save_chunk(chunk)

        return chunk

    def _backend_worker_loop(self):
        while not self.backend_stop_event.is_set():
            try:
                job = self.backend_job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                chunk = self._run_da3_job(job)
                opt_image_ids = self._optimize_latest_chunks()
                self._save_current_keyframes()
                self._save_manifest()

                self.backend_result_queue.put(
                    DA3BackendResult(
                        chunk_id=chunk.chunk_id,
                        image_ids=chunk.image_ids,
                        optimized_image_ids=opt_image_ids,
                    )
                )
            except Exception as exc:
                self.backend_result_queue.put(
                    DA3BackendResult(
                        chunk_id=job.chunk_id,
                        image_ids=job.image_ids,
                        error=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                )
            finally:
                self.backend_job_queue.task_done()

    def _collect_backend_results(self, output: MappingProcessResult):
        while True:
            try:
                result = self.backend_result_queue.get_nowait()
            except queue.Empty:
                break

            if result.error is not None:
                output.backend_error = result.error
            else:
                output.da3_ran = True
                output.new_chunk_id = result.chunk_id
                output.completed_chunk_ids.append(result.chunk_id)
                output.optimized_image_ids.extend(result.optimized_image_ids)

            self.backend_result_queue.task_done()

        if output.optimized_image_ids:
            output.optimization_ran = True
            output.optimized_image_ids = sorted(set(output.optimized_image_ids))

    # ---------------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------------
    def _optimize_latest_chunks(self) -> List[int]:
        with self.state_lock:
            if len(self.chunks) == 0:
                return []

            chunks_to_optimize = list(self.chunks[-self.optimizer_num_chunks :])
            optimized_pose_c2w = {
                int(k): np.asarray(v, dtype=np.float64).copy() for k, v in self.optimized_pose_c2w.items()
            }
            optimized_chunk_scales = dict(self.optimized_chunk_scales)

        anchor_chunk = chunks_to_optimize[0]

        # add poses
        pose_chunks = [self._chunk_record_to_pose_chunk(c, optimized_chunk_scales) for c in chunks_to_optimize]

        # add priors
        scale_priors = []
        pose_priors = []

        # the scale of other chunks couldn't be too different from 1
        for chunk in chunks_to_optimize:
            scale_priors.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "scale": 1.0,
                    "weight": self.scale_be_one_prior_weight,
                }
            )

        local2world = anchor_chunk.extrinsics_list[0]
        if anchor_chunk.image_ids[0] in optimized_pose_c2w:
            local2world = optimized_pose_c2w[anchor_chunk.image_ids[0]]

        result = self.optimizer.optimize(
            chunks=pose_chunks,
            pose_priors=pose_priors,
            scale_priors=scale_priors,
            fix_first_pose_identity=True,
            fix_first_chunk_scale_one=False,
        )

        optimized_image_ids = sorted(result.pose_matrices.keys())

        with self.state_lock:
            for image_id, pose_c2l in result.pose_matrices.items():
                image_id = int(image_id)
                # convert the pose to world coordinate
                self.updated_keyframes.add(image_id)

                pose_c2l = np.asarray(pose_c2l, dtype=np.float64)
                pose_c2w = np.dot(local2world, pose_c2l)
                pose_w2c = np.linalg.inv(pose_c2w)

                self.optimized_pose_c2w[image_id] = pose_c2w

                if int(image_id) in self.keyframes:
                    kf = self.keyframes[image_id]
                    kf.pose_c2w = pose_c2w
                    kf.pose_w2c = pose_w2c

        with self.state_lock:
            for chunk in chunks_to_optimize:
                if chunk.chunk_id in result.chunk_scales:
                    chunk.scale = float(result.chunk_scales[chunk.chunk_id])
                    chunk.initial_error = result.initial_error
                    chunk.final_error = result.final_error
                    self.optimized_chunk_scales[chunk.chunk_id] = chunk.scale

        return optimized_image_ids

    def _chunk_record_to_pose_chunk(
        self,
        chunk: DA3ChunkRecord,
        optimized_chunk_scales: Optional[Dict[int, float]] = None,
    ):
        scale = float(chunk.scale)
        optimized_chunk_scales = optimized_chunk_scales or self.optimized_chunk_scales
        if chunk.chunk_id in optimized_chunk_scales:
            scale = optimized_chunk_scales[chunk.chunk_id]
        if not np.isfinite(scale) or scale <= 1e-4:
            print(f"[WARN] Chunk {chunk.chunk_id} has invalid scale {scale}, " "resetting to 1.0")
            scale = 1.0

        return PoseChunk(
            chunk_id=chunk.chunk_id,
            image_ids=chunk.image_ids,
            poses=chunk.extrinsics_list,
            scale=scale,
            weight=1.0,
        )

    def _make_anchor_pose_priors(self, anchor_chunk: DA3ChunkRecord):
        priors = []

        for image_id in anchor_chunk.image_ids:
            if image_id not in self.optimized_pose_c2w:
                continue

            priors.append(
                ImagePosePrior(
                    image_id=int(image_id),
                    pose=self.optimized_pose_c2w[image_id],
                    weight=self.anchor_prior_weight,
                )
            )

        return priors

    def get_trajectory_snapshot(self):
        with self.state_lock:
            return {
                "poses_c2w": {
                    int(k): np.asarray(v, dtype=np.float64).copy() for k, v in self.optimized_pose_c2w.items()
                },
                "frame_id": int(self.frame_id),
                "latest_keyframe_id": int(self.next_keyframe_id - 1) if self.next_keyframe_id > 0 else None,
                "latest_chunk_id": int(self.chunks[-1].chunk_id) if len(self.chunks) > 0 else None,
            }

    # ---------------------------------------------------------------------
    # Saving
    # ---------------------------------------------------------------------

    def _save_keyframe_rgb(self, kf: KeyframeRecord):
        path = self.keyframe_dir / f"kf_{kf.image_id:06d}_rgb.png"
        cv2.imwrite(str(path), kf.image_bgr)

    def _save_all_complete_keyframes(self):
        with self.state_lock:
            keyframes = list(self.keyframes.values())

        for kf in keyframes:
            self._save_keyframe(kf)

    def _save_current_keyframes(self):
        with self.state_lock:
            keyframes = [
                self.keyframes[image_id] for image_id in sorted(self.updated_keyframes) if image_id in self.keyframes
            ]
            self.updated_keyframes.clear()

        for kf in keyframes:
            self._save_keyframe(kf)

    def _save_keyframe(self, kf: KeyframeRecord):
        self._save_keyframe_rgb(kf)

        prefix = self.keyframe_dir / f"kf_{kf.image_id:06d}"

        if kf.depth is not None:
            np.save(str(prefix) + "_depth.npy", kf.depth.astype(np.float32))

            if self.save_depth_png_preview:
                depth_png = self._depth_to_u8(kf.depth)
                cv2.imwrite(str(prefix) + "_depth_vis.png", depth_png)

        if kf.intrinsics is not None:
            np.save(str(prefix) + "_intrinsics.npy", kf.intrinsics.astype(np.float64))

        if kf.pose_c2w is not None:
            np.save(str(prefix) + "_pose_c2w.npy", kf.pose_c2w.astype(np.float64))

        if kf.pose_w2c is not None:
            np.save(str(prefix) + "_pose_w2c.npy", kf.pose_w2c.astype(np.float64))

        meta = {
            "image_id": kf.image_id,
            "source_frame_id": kf.source_frame_id,
            "last_chunk_id": kf.last_chunk_id,
            "has_depth": kf.depth is not None,
            "has_intrinsics": kf.intrinsics is not None,
            "has_pose": kf.pose_c2w is not None,
            "rgb_path": f"kf_{kf.image_id:06d}_rgb.png",
            "depth_path": f"kf_{kf.image_id:06d}_depth.npy" if kf.depth is not None else None,
            "intrinsics_path": f"kf_{kf.image_id:06d}_intrinsics.npy" if kf.intrinsics is not None else None,
            "pose_c2w_path": f"kf_{kf.image_id:06d}_pose_c2w.npy" if kf.pose_c2w is not None else None,
            "pose_w2c_path": f"kf_{kf.image_id:06d}_pose_w2c.npy" if kf.pose_w2c is not None else None,
        }

        with open(str(prefix) + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _save_chunk(self, chunk: DA3ChunkRecord):
        chunk_path = self.chunk_dir / f"chunk_{chunk.chunk_id:06d}.json"

        meta = {
            "chunk_id": chunk.chunk_id,
            "image_ids": chunk.image_ids,
            "scale": chunk.scale,
            "initial_error": chunk.initial_error,
            "final_error": chunk.final_error,
        }

        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _save_manifest(self):
        with self.state_lock:
            keyframes = [self.keyframes[image_id] for image_id in sorted(self.keyframes.keys())]
            chunks = list(self.chunks)

            manifest = {
                "window_size": self.window_size,
                "da3_stride_new_keyframes": self.da3_stride_new_keyframes,
                "optimizer_num_chunks": self.optimizer_num_chunks,
                "num_keyframes": len(keyframes),
                "num_chunks": len(chunks),
                "keyframes": [],
                "chunks": [],
            }

            for kf in keyframes:
                manifest["keyframes"].append(
                    {
                        "image_id": kf.image_id,
                        "source_frame_id": kf.source_frame_id,
                        "last_chunk_id": kf.last_chunk_id,
                        "has_depth": kf.depth is not None,
                        "has_intrinsics": kf.intrinsics is not None,
                        "has_pose": kf.pose_c2w is not None,
                        "rgb_path": f"keyframes/kf_{kf.image_id:06d}_rgb.png",
                        "depth_path": f"keyframes/kf_{kf.image_id:06d}_depth.npy" if kf.depth is not None else None,
                        "intrinsics_path": f"keyframes/kf_{kf.image_id:06d}_intrinsics.npy"
                        if kf.intrinsics is not None
                        else None,
                        "pose_c2w_path": f"keyframes/kf_{kf.image_id:06d}_pose_c2w.npy"
                        if kf.pose_c2w is not None
                        else None,
                        "pose_w2c_path": f"keyframes/kf_{kf.image_id:06d}_pose_w2c.npy"
                        if kf.pose_w2c is not None
                        else None,
                    }
                )

            for chunk in chunks:
                manifest["chunks"].append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "image_ids": chunk.image_ids,
                        "scale": chunk.scale,
                        "initial_error": chunk.initial_error,
                        "final_error": chunk.final_error,
                    }
                )

        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------

    @staticmethod
    def _as_4x4(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T)

        if T.shape == (4, 4):
            return T.astype(np.float64)

        if T.shape == (3, 4):
            T44 = np.eye(4, dtype=np.float64)
            T44[:3, :4] = T.astype(np.float64)
            return T44

        raise ValueError(f"Expected pose shape (3, 4) or (4, 4), got {T.shape}")

    @staticmethod
    def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None")

        img = np.asarray(image)

        if img.ndim == 2:
            if img.dtype != np.uint8:
                if np.nanmax(img) <= 1.0:
                    img = img * 255.0
                img = np.clip(img, 0, 255).astype(np.uint8)

            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3 and img.shape[2] == 3:
            if img.dtype != np.uint8:
                if np.nanmax(img) <= 1.0:
                    img = img * 255.0
                img = np.clip(img, 0, 255).astype(np.uint8)

            # Assumes OpenCV BGR input.
            return img.copy()

        raise ValueError(f"Unsupported image shape: {img.shape}")

    @staticmethod
    def _depth_to_u8(depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)

        valid = np.isfinite(depth)
        if not np.any(valid):
            return np.zeros(depth.shape, dtype=np.uint8)

        v = depth[valid]
        d_min = np.percentile(v, 2)
        d_max = np.percentile(v, 98)

        depth_norm = np.clip(depth, d_min, d_max)
        depth_norm = (depth_norm - d_min) / (d_max - d_min + 1e-8)

        # Near brighter, far darker.
        depth_norm = 1.0 - depth_norm

        return (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)


# python da3_slam/da3_streaming.py
if __name__ == "__main__":
    flow_processor = OpticalFlowKeyframeProcessor(
        min_feature_distance=10,
        keyframe_pixel_threshold=50.0,
        min_tracked_features=150,
    )

    da3_client = DepthAnything3(
        triton_url="0.0.0.0:8001",
        expected_num_images=None,
    )

    pose_optimizer = DA3ChunkPoseGraphOptimizer(
        input_pose_is_w2c=True,  # DA3 extrinsics are usually w2c
        relative_rotation_sigma=0.03,
        relative_translation_sigma=0.05,
        max_iterations=100,
    )

    pipeline = DA3StreamingMappingPipeline(
        flow_processor=flow_processor,
        da3_client=da3_client,
        pose_graph_optimizer=pose_optimizer,
        output_dir="output_da3_map",
        window_size=10,  # N
        da3_stride_new_keyframes=5,  # M
        optimizer_num_chunks=3,  # K
    )

    traj_vis = MatplotlibTrajectoryVisualizer(
        mode="3d",
        update_hz=5.0,
        window_title="DA3-SLAM 3D Trajectory",
        draw_camera_direction=True,
    )
    # traj_vis = MatplotlibTrajectoryVisualizer(
    #     mode="2d",
    #     plane="xz",
    #     update_hz=5.0,
    #     window_title="DA3-SLAM Top View",
    # )
    traj_vis.start()

    import glob

    image_files = glob.glob("data/cam0-20260427T031458Z-3-001/cam0/data/*.png")
    image_files.sort()

    quit_requested = False

    try:
        for image_file in image_files:
            image = cv2.imread(image_file)

            result = pipeline.process_image(image)
            # Update trajectory window
            traj_vis.update_from_pipeline(pipeline)

            if result.is_keyframe:
                print(f"New keyframe: {result.keyframe_id}")

            if result.da3_scheduled:
                print(f"DA3 chunk scheduled: {result.scheduled_chunk_id}")

            if result.da3_ran:
                print(f"DA3 chunk completed: {result.new_chunk_id}")

            if result.optimization_ran:
                print(f"Optimized images: {result.optimized_image_ids}")

            if result.backend_error:
                print(f"DA3 backend error: {result.backend_error}")

            vis = visualize_optical_flow_result(image, result.flow_result)

            cv2.imshow("optical flow tracks", vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                quit_requested = True
                break
    finally:
        pipeline.close(wait=not quit_requested, drain=not quit_requested)
        cv2.destroyAllWindows()
        traj_vis.stop()
