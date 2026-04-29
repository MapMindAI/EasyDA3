from __future__ import annotations

import queue
import threading
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

try:
    from ..backend.da3_pose_graph_optimizer import ImagePosePrior, PoseChunk
    from ..logging_utils import get_logger
    from .records import (
        DA3BackendJob,
        DA3BackendResult,
        DA3ChunkRecord,
        KeyframeRecord,
        MappingProcessResult,
    )
    from .storage import StreamingMapStorage
    from .utils import as_4x4, estimate_pose_c2w_cur, project_depth_to_frame, to_bgr_uint8
except ImportError:
    from backend.da3_pose_graph_optimizer import ImagePosePrior, PoseChunk
    from logging_utils import get_logger
    from streaming.records import (
        DA3BackendJob,
        DA3BackendResult,
        DA3ChunkRecord,
        KeyframeRecord,
        MappingProcessResult,
    )
    from streaming.storage import StreamingMapStorage
    from streaming.utils import as_4x4, estimate_pose_c2w_cur, project_depth_to_frame, to_bgr_uint8


logger = get_logger(__name__)


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
        enable_frontend_pnp: bool = True,
        pnp_min_inliers: int = 12,
        pnp_reprojection_error_px: float = 4.0,
        projected_depth_splat_radius: int = 1,
        projected_depth_min_valid_pixels: int = 64,
    ):
        """
        Streaming visual mapping pipeline.

        Main thread:
            Runs optical flow and keyframe bookkeeping.

        Backend thread:
            Runs DA3, pose-graph optimization, and disk persistence.
        """

        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if da3_stride_new_keyframes < 1:
            raise ValueError("da3_stride_new_keyframes must be >= 1")
        if optimizer_num_chunks < 1:
            raise ValueError("optimizer_num_chunks must be >= 1")
        if pnp_min_inliers < 4:
            raise ValueError("pnp_min_inliers must be >= 4")
        if projected_depth_min_valid_pixels < 0:
            raise ValueError("projected_depth_min_valid_pixels must be >= 0")

        self.state_lock = threading.Lock()
        self.flow_processor = flow_processor
        self.da3_client = da3_client
        self.optimizer = pose_graph_optimizer

        self.window_size = int(window_size)
        self.da3_stride_new_keyframes = int(da3_stride_new_keyframes)
        self.optimizer_num_chunks = int(optimizer_num_chunks)
        self.anchor_prior_weight = float(anchor_prior_weight)
        self.scale_be_one_prior_weight = float(1e1)
        self.save_depth_png_preview = bool(save_depth_png_preview)
        self.enable_frontend_pnp = bool(enable_frontend_pnp)
        self.pnp_min_inliers = int(pnp_min_inliers)
        self.pnp_reprojection_error_px = float(pnp_reprojection_error_px)
        self.projected_depth_splat_radius = int(projected_depth_splat_radius)
        self.projected_depth_min_valid_pixels = int(projected_depth_min_valid_pixels)
        self.min_valid_chunk_scale = 1e-4
        self.max_valid_chunk_scale = 1e4

        self.storage = StreamingMapStorage(
            output_dir=output_dir,
            save_depth_png_preview=save_depth_png_preview,
        )
        self.output_dir = self.storage.output_dir
        self.keyframe_dir = self.storage.keyframe_dir
        self.chunk_dir = self.storage.chunk_dir

        self.frame_id = -1
        self.next_keyframe_id = 0
        self.next_chunk_id = 0

        self.sliding_keyframe_ids = deque(maxlen=self.window_size)
        self.keyframes: Dict[int, KeyframeRecord] = {}
        self.source_frame_to_keyframe_id: Dict[int, int] = {}
        self.chunks: List[DA3ChunkRecord] = []

        self.new_keyframes_since_last_da3 = 0

        self.optimized_pose_c2w: Dict[int, np.ndarray] = {}
        self.frontend_pose_c2w: Dict[int, np.ndarray] = {}
        self.latest_frame_pose_c2w: Optional[np.ndarray] = None
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

    def process_image(self, image: np.ndarray) -> MappingProcessResult:
        """
        Process one incoming image.

        DA3 and optimization are scheduled asynchronously. The returned result
        contains immediate optical-flow state plus any finished backend work
        collected at the start of this call.
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
        self._estimate_pose_from_flow(flow_result, output)

        if not flow_result.is_keyframe:
            return output

        keyframe_id = self._add_keyframe(image, source_frame_id=frame_id)
        output.keyframe_id = keyframe_id
        self._attach_frontend_pose_to_keyframe(keyframe_id, output)

        if output.pose_reference_keyframe_id is not None:
            self._project_depth_between_keyframes(
                source_keyframe_id=output.pose_reference_keyframe_id,
                target_keyframe_id=keyframe_id,
                output=output,
            )

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

    def _add_keyframe(self, image: np.ndarray, source_frame_id: int) -> int:
        image_bgr = to_bgr_uint8(image)

        with self.state_lock:
            image_id = self.next_keyframe_id
            self.next_keyframe_id += 1

            record = KeyframeRecord(
                image_id=image_id,
                source_frame_id=source_frame_id,
                image_bgr=image_bgr.copy(),
            )

            self.keyframes[image_id] = record
            self.source_frame_to_keyframe_id[int(source_frame_id)] = image_id
            self.sliding_keyframe_ids.append(image_id)
            self.new_keyframes_since_last_da3 += 1

        self.storage.save_keyframe_rgb(record)

        return image_id

    def _should_run_da3(self) -> bool:
        with self.state_lock:
            if len(self.sliding_keyframe_ids) < self.window_size:
                return False

            if self.new_keyframes_since_last_da3 < self.da3_stride_new_keyframes:
                return False

            return True

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
        depth_conf_list_raw = da3_result.get("depth_conf_list", None)
        if depth_conf_list_raw is None:
            depth_conf_list = [np.ones_like(d, dtype=np.float32) for d in depth_list]
        else:
            depth_conf_list = [np.asarray(c, dtype=np.float32) for c in depth_conf_list_raw]
        intrinsics_list = [np.asarray(k, dtype=np.float32) for k in da3_result["intrinsics_list"]]
        extrinsics_list = [as_4x4(np.asarray(e, dtype=np.float32)) for e in da3_result["extrinsics_list"]]

        if not (
            len(depth_list)
            == len(depth_conf_list)
            == len(intrinsics_list)
            == len(extrinsics_list)
            == len(job.image_ids)
        ):
            raise RuntimeError("DA3 output count does not match input keyframe count.")

        chunk = DA3ChunkRecord(
            chunk_id=job.chunk_id,
            image_ids=job.image_ids,
            depth_list=depth_list,
            depth_conf_list=depth_conf_list,
            intrinsics_list=intrinsics_list,
            extrinsics_list=extrinsics_list,
            scale=1.0,
        )

        with self.state_lock:
            self.chunks.append(chunk)

            for local_idx, image_id in enumerate(job.image_ids):
                keyframe = self.keyframes.get(image_id)
                if keyframe is None:
                    continue

                keyframe.depth = depth_list[local_idx]
                keyframe.depth_conf = depth_conf_list[local_idx]
                keyframe.intrinsics = intrinsics_list[local_idx]
                keyframe.da3_local_w2c = extrinsics_list[local_idx]
                keyframe.last_chunk_id = job.chunk_id
                self.updated_keyframes.add(image_id)

        self.storage.save_chunk(chunk)

        return chunk

    def _backend_worker_loop(self):
        while not self.backend_stop_event.is_set():
            try:
                job = self.backend_job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                chunk = self._run_da3_job(job)
                optimized_image_ids = self._optimize_latest_chunks()
                self._save_current_keyframes()
                self._save_manifest()

                self.backend_result_queue.put(
                    DA3BackendResult(
                        chunk_id=chunk.chunk_id,
                        image_ids=chunk.image_ids,
                        optimized_image_ids=optimized_image_ids,
                    )
                )
            except Exception as exc:
                logger.exception("DA3 backend job %s failed", job.chunk_id)
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

    def _estimate_pose_from_flow(self, flow_result, output: MappingProcessResult) -> None:
        if not self.enable_frontend_pnp:
            return

        tracks = getattr(flow_result, "tracks", None)
        if tracks is None or len(tracks) < 6:
            return

        reference_source_frame_id = int(flow_result.keyframe_id)
        with self.state_lock:
            reference_keyframe_id = self.source_frame_to_keyframe_id.get(reference_source_frame_id)

        if reference_keyframe_id is None:
            return

        if not self._ensure_keyframe_depth_available(reference_keyframe_id):
            return

        geometry = self._get_keyframe_geometry_for_pnp(reference_keyframe_id)
        if geometry is None:
            return

        depth_kf, intrinsics_kf, pose_c2w_kf, depth_scale = geometry

        try:
            pose_c2w_cur, info = estimate_pose_c2w_cur(
                tracks=tracks,
                depth_kf=depth_kf,
                intrinsics_kf=intrinsics_kf,
                pose_c2w_kf=pose_c2w_kf,
                intrinsics_cur=intrinsics_kf,
                reprojection_error_px=self.pnp_reprojection_error_px,
                min_inliers=self.pnp_min_inliers,
                return_info=True,
            )
        except (RuntimeError, ValueError) as exc:
            output.pose_estimation_error = str(exc)
            return

        output.estimated_pose_c2w = pose_c2w_cur
        output.estimated_pose_w2c = np.linalg.inv(pose_c2w_cur)
        output.pose_reference_keyframe_id = int(reference_keyframe_id)
        output.pose_reference_source_frame_id = reference_source_frame_id
        info["reference_keyframe_id"] = int(reference_keyframe_id)
        info["reference_source_frame_id"] = reference_source_frame_id
        info["reference_depth_scale"] = float(depth_scale)
        output.pose_estimation_info = info
        with self.state_lock:
            self.latest_frame_pose_c2w = np.asarray(pose_c2w_cur, dtype=np.float64).copy()

    def _ensure_keyframe_depth_available(self, keyframe_id: int) -> bool:
        with self.state_lock:
            keyframe = self.keyframes.get(keyframe_id)
            if keyframe is None:
                return False

            if keyframe.depth is not None and keyframe.intrinsics is not None and keyframe.pose_c2w is not None:
                return True

            can_project_to_keyframe = keyframe.pose_c2w is not None

        if not can_project_to_keyframe:
            return False

        source_keyframe_id = self._find_latest_depth_source_before(keyframe_id)
        if source_keyframe_id is None:
            return False

        return self._project_depth_between_keyframes(
            source_keyframe_id=source_keyframe_id,
            target_keyframe_id=keyframe_id,
            output=None,
        )

    def _find_latest_depth_source_before(self, target_keyframe_id: int) -> Optional[int]:
        with self.state_lock:
            for keyframe_id in sorted(self.keyframes.keys(), reverse=True):
                if keyframe_id >= target_keyframe_id:
                    continue

                keyframe = self.keyframes[keyframe_id]
                if keyframe.depth is None or keyframe.intrinsics is None or keyframe.pose_c2w is None:
                    continue

                return int(keyframe_id)

        return None

    def _get_keyframe_geometry_for_pnp(self, keyframe_id: int):
        with self.state_lock:
            keyframe = self.keyframes.get(keyframe_id)
            if keyframe is None:
                return None

            if keyframe.depth is None or keyframe.intrinsics is None or keyframe.pose_c2w is None:
                return None

            depth_scale = self._keyframe_depth_scale_locked(keyframe)
            depth = np.asarray(keyframe.depth, dtype=np.float32).copy()
            intrinsics = np.asarray(keyframe.intrinsics, dtype=np.float64).copy()
            pose_c2w = np.asarray(keyframe.pose_c2w, dtype=np.float64).copy()

        if depth_scale != 1.0:
            depth = depth * np.float32(depth_scale)

        return depth, intrinsics, pose_c2w, depth_scale

    def _keyframe_depth_scale_locked(self, keyframe: KeyframeRecord) -> float:
        if keyframe.last_chunk_id is None:
            return 1.0

        scale = self.optimized_chunk_scales.get(keyframe.last_chunk_id)
        if scale is None:
            for chunk in reversed(self.chunks):
                if chunk.chunk_id == keyframe.last_chunk_id:
                    scale = chunk.scale
                    break

        if scale is None:
            return 1.0

        scale = float(scale)
        if not np.isfinite(scale) or scale < self.min_valid_chunk_scale or scale > self.max_valid_chunk_scale:
            return 1.0

        return scale

    def _attach_frontend_pose_to_keyframe(self, keyframe_id: int, output: MappingProcessResult) -> None:
        if output.estimated_pose_c2w is None:
            return

        pose_c2w = np.asarray(output.estimated_pose_c2w, dtype=np.float64)
        pose_w2c = np.linalg.inv(pose_c2w)

        with self.state_lock:
            keyframe = self.keyframes.get(keyframe_id)
            if keyframe is None:
                return

            keyframe.pose_c2w = pose_c2w
            keyframe.pose_w2c = pose_w2c
            self.frontend_pose_c2w[int(keyframe_id)] = pose_c2w
            self.updated_keyframes.add(int(keyframe_id))

        self.storage.save_keyframe(keyframe)

    def _project_depth_between_keyframes(
        self,
        source_keyframe_id: int,
        target_keyframe_id: int,
        output: Optional[MappingProcessResult],
    ) -> bool:
        source_geometry = self._get_keyframe_geometry_for_pnp(source_keyframe_id)
        if source_geometry is None:
            return False

        depth_src, intrinsics_src, pose_c2w_src, _ = source_geometry

        with self.state_lock:
            target = self.keyframes.get(target_keyframe_id)
            if target is None or target.pose_c2w is None:
                return False

            if target.depth is not None and target.intrinsics is not None:
                return True

            target_pose_c2w = np.asarray(target.pose_c2w, dtype=np.float64).copy()
            target_intrinsics = (
                np.asarray(target.intrinsics, dtype=np.float64).copy()
                if target.intrinsics is not None
                else intrinsics_src.copy()
            )
            target_shape = target.image_bgr.shape[:2]

        try:
            projected_depth = project_depth_to_frame(
                depth_src=depth_src,
                intrinsics_src=intrinsics_src,
                pose_c2w_src=pose_c2w_src,
                intrinsics_dst=target_intrinsics,
                pose_c2w_dst=target_pose_c2w,
                dst_shape=target_shape,
                splat_radius=self.projected_depth_splat_radius,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            logger.debug(
                "Depth projection %s -> %s failed: %s",
                source_keyframe_id,
                target_keyframe_id,
                exc,
            )
            return False

        valid_pixels = int(np.count_nonzero(np.isfinite(projected_depth) & (projected_depth > 0.0)))
        if valid_pixels < self.projected_depth_min_valid_pixels:
            logger.debug(
                "Depth projection %s -> %s produced too few pixels: %s",
                source_keyframe_id,
                target_keyframe_id,
                valid_pixels,
            )
            return False

        with self.state_lock:
            target = self.keyframes.get(target_keyframe_id)
            if target is None:
                return False

            target.depth = projected_depth.astype(np.float32)
            target.depth_conf = np.where(
                np.isfinite(projected_depth) & (projected_depth > 0.0),
                1.0,
                0.0,
            ).astype(np.float32)
            target.intrinsics = target_intrinsics
            self.updated_keyframes.add(int(target_keyframe_id))

        if output is not None:
            output.projected_depth_keyframe_id = int(target_keyframe_id)
            output.projected_depth_source_keyframe_id = int(source_keyframe_id)
            output.projected_depth_valid_pixels = valid_pixels

        self.storage.save_keyframe(target)
        return True

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
        pose_chunks = [self._chunk_record_to_pose_chunk(c, optimized_chunk_scales) for c in chunks_to_optimize]
        scale_priors = [
            {
                "chunk_id": chunk.chunk_id,
                "scale": 1.0,
                "weight": self.scale_be_one_prior_weight,
            }
            for chunk in chunks_to_optimize
        ]

        local2world = anchor_chunk.extrinsics_list[0]
        if anchor_chunk.image_ids[0] in optimized_pose_c2w:
            local2world = optimized_pose_c2w[anchor_chunk.image_ids[0]]

        result = self.optimizer.optimize(
            chunks=pose_chunks,
            pose_priors=[],
            scale_priors=scale_priors,
            fix_first_pose_identity=True,
            fix_first_chunk_scale_one=False,
        )

        optimized_image_ids = sorted(result.pose_matrices.keys())

        with self.state_lock:
            for image_id, pose_c2l in result.pose_matrices.items():
                image_id = int(image_id)
                self.updated_keyframes.add(image_id)

                pose_c2l = np.asarray(pose_c2l, dtype=np.float64)
                pose_c2w = np.dot(local2world, pose_c2l)
                pose_w2c = np.linalg.inv(pose_c2w)

                self.optimized_pose_c2w[image_id] = pose_c2w

                if image_id in self.keyframes:
                    keyframe = self.keyframes[image_id]
                    keyframe.pose_c2w = pose_c2w
                    keyframe.pose_w2c = pose_w2c

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
            logger.warning("Chunk %s has invalid scale %s; resetting to 1.0", chunk.chunk_id, scale)
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
            poses_c2w = {
                int(k): np.asarray(v, dtype=np.float64).copy() for k, v in self.frontend_pose_c2w.items()
            }
            poses_c2w.update(
                {int(k): np.asarray(v, dtype=np.float64).copy() for k, v in self.optimized_pose_c2w.items()}
            )

            return {
                "poses_c2w": poses_c2w,
                "current_pose_c2w": (
                    None
                    if self.latest_frame_pose_c2w is None
                    else np.asarray(self.latest_frame_pose_c2w, dtype=np.float64).copy()
                ),
                "frame_id": int(self.frame_id),
                "latest_keyframe_id": int(self.next_keyframe_id - 1) if self.next_keyframe_id > 0 else None,
                "latest_chunk_id": int(self.chunks[-1].chunk_id) if len(self.chunks) > 0 else None,
            }

    def _save_all_complete_keyframes(self):
        with self.state_lock:
            keyframes = list(self.keyframes.values())

        for keyframe in keyframes:
            self.storage.save_keyframe(keyframe)

    def _save_current_keyframes(self):
        with self.state_lock:
            keyframes = [
                self.keyframes[image_id] for image_id in sorted(self.updated_keyframes) if image_id in self.keyframes
            ]
            self.updated_keyframes.clear()

        for keyframe in keyframes:
            self.storage.save_keyframe(keyframe)

    def _save_manifest(self):
        with self.state_lock:
            keyframes = [self.keyframes[image_id] for image_id in sorted(self.keyframes.keys())]
            chunks = list(self.chunks)

        self.storage.save_manifest(
            window_size=self.window_size,
            da3_stride_new_keyframes=self.da3_stride_new_keyframes,
            optimizer_num_chunks=self.optimizer_num_chunks,
            keyframes=keyframes,
            chunks=chunks,
        )
