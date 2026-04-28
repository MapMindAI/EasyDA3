from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

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
class MappingProcessResult:
    frame_id: int
    is_keyframe: bool
    keyframe_id: Optional[int]

    da3_ran: bool = False
    new_chunk_id: Optional[int] = None

    optimization_ran: bool = False
    optimized_image_ids: List[int] = field(default_factory=list)

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

        self.flow_processor = flow_processor
        self.da3_client = da3_client
        self.optimizer = pose_graph_optimizer

        self.window_size = int(window_size)
        self.da3_stride_new_keyframes = int(da3_stride_new_keyframes)
        self.optimizer_num_chunks = int(optimizer_num_chunks)
        self.anchor_prior_weight = float(anchor_prior_weight)
        self.scale_be_one_prior_weight = float(1e2)
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
            3. If window is full and enough new keyframes exist, run DA3.
            4. After DA3, optimize latest K chunks.
            5. Save keyframe RGB/depth/intrinsics/pose to disk.

        Returns:
            MappingProcessResult
        """

        self.frame_id += 1

        flow_result = self.flow_processor.process(image)

        output = MappingProcessResult(
            frame_id=self.frame_id,
            is_keyframe=bool(flow_result.is_keyframe),
            keyframe_id=None,
            flow_result=flow_result,
        )

        if not flow_result.is_keyframe:
            return output

        keyframe_id = self._add_keyframe(image)
        output.keyframe_id = keyframe_id

        if self._should_run_da3():
            chunk = self._run_da3_on_current_window()
            output.da3_ran = True
            output.new_chunk_id = chunk.chunk_id

            opt_image_ids = self._optimize_latest_chunks()
            output.optimization_ran = len(opt_image_ids) > 0
            output.optimized_image_ids = opt_image_ids

            self._save_all_complete_keyframes()
            self._save_manifest()

        return output

    # ---------------------------------------------------------------------
    # Keyframe handling
    # ---------------------------------------------------------------------

    def _add_keyframe(self, image: np.ndarray) -> int:
        image_bgr = self._to_bgr_uint8(image)

        image_id = self.next_keyframe_id
        self.next_keyframe_id += 1

        record = KeyframeRecord(
            image_id=image_id,
            source_frame_id=self.frame_id,
            image_bgr=image_bgr.copy(),
        )

        self.keyframes[image_id] = record
        self.sliding_keyframe_ids.append(image_id)
        self.new_keyframes_since_last_da3 += 1

        self._save_keyframe_rgb(record)

        return image_id

    def _should_run_da3(self) -> bool:
        if len(self.sliding_keyframe_ids) < self.window_size:
            return False

        if self.new_keyframes_since_last_da3 < self.da3_stride_new_keyframes:
            return False

        return True

    # ---------------------------------------------------------------------
    # DA3 chunk processing
    # ---------------------------------------------------------------------

    def _run_da3_on_current_window(self) -> DA3ChunkRecord:
        image_ids = list(self.sliding_keyframe_ids)
        images = [self.keyframes[i].image_bgr for i in image_ids]

        da3_result = self.da3_client.run(images)

        depth_list = [np.asarray(d, dtype=np.float32) for d in da3_result["depth_list"]]
        intrinsics_list = [np.asarray(k, dtype=np.float32) for k in da3_result["intrinsics_list"]]
        extrinsics_list = [self._as_4x4(np.asarray(e, dtype=np.float32)) for e in da3_result["extrinsics_list"]]

        if not (len(depth_list) == len(intrinsics_list) == len(extrinsics_list) == len(image_ids)):
            raise RuntimeError("DA3 output count does not match input keyframe count.")

        chunk_id = self.next_chunk_id
        self.next_chunk_id += 1

        chunk = DA3ChunkRecord(
            chunk_id=chunk_id,
            image_ids=image_ids,
            depth_list=depth_list,
            intrinsics_list=intrinsics_list,
            extrinsics_list=extrinsics_list,
            scale=1.0,
        )

        self.chunks.append(chunk)

        for local_idx, image_id in enumerate(image_ids):
            kf = self.keyframes[image_id]
            kf.depth = depth_list[local_idx]
            kf.intrinsics = intrinsics_list[local_idx]
            kf.da3_local_w2c = extrinsics_list[local_idx]
            kf.last_chunk_id = chunk_id

        self.new_keyframes_since_last_da3 = 0

        self._save_chunk(chunk)

        return chunk

    # ---------------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------------

    def _optimize_latest_chunks(self) -> List[int]:
        if len(self.chunks) == 0:
            return []

        chunks_to_optimize = self.chunks[-self.optimizer_num_chunks :]

        pose_chunks = [self._chunk_record_to_pose_chunk(c) for c in chunks_to_optimize]

        anchor_chunk = chunks_to_optimize[0]

        pose_priors = self._make_anchor_pose_priors(anchor_chunk)
        scale_priors = []

        is_first_optimization = len(self.optimized_pose_c2w) == 0

        if is_first_optimization:
            # First ever local map:
            #   first image pose = identity
            #   first chunk scale = 1
            fix_first_pose_identity = True
            fix_first_chunk_scale_one = True
        else:
            # Later local maps:
            #   first chunk is fixed using strong pose priors.
            #
            # The scale of the anchor chunk is effectively fixed because all
            # poses inside that chunk are strongly constrained by priors.
            fix_first_pose_identity = False
            fix_first_chunk_scale_one = False
            anchor_scale = self._sanitize_chunk_scale(anchor_chunk.scale, default=1.0)

            scale_priors.append(
                {
                    "chunk_id": anchor_chunk.chunk_id,
                    "scale": anchor_scale,
                    "weight": 1e2,
                }
            )

        # the scale of other chunks couldn't be too different from 1
        for chunk in chunks_to_optimize:
            scale_priors.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "scale": 1.0,
                    "weight": self.scale_be_one_prior_weight,
                }
            )

        result = self.optimizer.optimize(
            chunks=pose_chunks,
            pose_priors=pose_priors,
            scale_priors=scale_priors,
            fix_first_pose_identity=fix_first_pose_identity,
            fix_first_chunk_scale_one=fix_first_chunk_scale_one,
        )

        optimized_image_ids = sorted(result.pose_matrices.keys())

        for image_id, pose_c2w in result.pose_matrices.items():
            pose_c2w = np.asarray(pose_c2w, dtype=np.float64)
            pose_w2c = np.linalg.inv(pose_c2w)

            self.optimized_pose_c2w[int(image_id)] = pose_c2w

            if int(image_id) in self.keyframes:
                kf = self.keyframes[int(image_id)]
                kf.pose_c2w = pose_c2w
                kf.pose_w2c = pose_w2c

        for chunk in chunks_to_optimize:
            if chunk.chunk_id in result.chunk_scales:
                chunk.scale = float(result.chunk_scales[chunk.chunk_id])
                chunk.initial_error = result.initial_error
                chunk.final_error = result.final_error
                self.optimized_chunk_scales[chunk.chunk_id] = chunk.scale

        return optimized_image_ids

    def _chunk_record_to_pose_chunk(self, chunk: DA3ChunkRecord):
        scale = float(chunk.scale)
        if not np.isfinite(scale) or scale <= 1e-4:
            print(f"[WARN] Chunk {chunk.chunk_id} has invalid scale {scale}, " "resetting to 1.0")
            scale = 1.0
        # print(f"{chunk.chunk_id} {scale}")

        return PoseChunk(
            chunk_id=chunk.chunk_id,
            image_ids=chunk.image_ids,
            poses=chunk.extrinsics_list,
            scale=chunk.scale,
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

    # ---------------------------------------------------------------------
    # Saving
    # ---------------------------------------------------------------------

    def _save_keyframe_rgb(self, kf: KeyframeRecord):
        path = self.keyframe_dir / f"kf_{kf.image_id:06d}_rgb.png"
        cv2.imwrite(str(path), kf.image_bgr)

    def _save_all_complete_keyframes(self):
        for kf in self.keyframes.values():
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
        manifest = {
            "window_size": self.window_size,
            "da3_stride_new_keyframes": self.da3_stride_new_keyframes,
            "optimizer_num_chunks": self.optimizer_num_chunks,
            "num_keyframes": len(self.keyframes),
            "num_chunks": len(self.chunks),
            "keyframes": [],
            "chunks": [],
        }

        for image_id in sorted(self.keyframes.keys()):
            kf = self.keyframes[image_id]

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

        for chunk in self.chunks:
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
        keyframe_pixel_threshold=25.0,
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
        optimizer_num_chunks=5,  # K
    )

    import glob

    image_files = glob.glob("data/cam0-20260427T031458Z-3-001/cam0/data/*.png")
    image_files.sort()

    for image_file in image_files:
        image = cv2.imread(image_file)

        result = pipeline.process_image(image)

        if result.is_keyframe:
            print(f"New keyframe: {result.keyframe_id}")

        if result.da3_ran:
            print(f"DA3 chunk created: {result.new_chunk_id}")

        if result.optimization_ran:
            print(f"Optimized images: {result.optimized_image_ids}")

        vis = visualize_optical_flow_result(image, result.flow_result)

        cv2.imshow("optical flow tracks", vis)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
