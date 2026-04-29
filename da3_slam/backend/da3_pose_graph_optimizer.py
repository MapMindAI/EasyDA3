# Copyright 2026 MapMind Inc. All rights reserved.
#
# GTSAM Python pose-graph optimizer
# a reusable optimizer class. I’ll assume DA3 chunk poses are local SE(3) camera poses, while each chunk
# has a scalar scale variable; the graph estimates global image poses plus chunk scales.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Optional, Tuple
import math

import numpy as np
import gtsam

try:
    from ..logging_utils import configure_logging, get_logger
except ImportError:
    try:
        from logging_utils import configure_logging, get_logger
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from logging_utils import configure_logging, get_logger


logger = get_logger(__name__)


@dataclass
class PoseChunk:
    chunk_id: int
    image_ids: Sequence[int]
    poses: Sequence[Any]  # each pose: gtsam.Pose3 or 4x4 numpy matrix
    scale: float = 1.0  # initial scale factor
    weight: float = 1.0  # confidence of this chunk's relative constraints


@dataclass
class ImagePosePrior:
    image_id: int
    pose: Any  # gtsam.Pose3 or 4x4 numpy matrix
    weight: float = 1.0  # larger means stronger prior


@dataclass
class ChunkScalePrior:
    chunk_id: int
    scale: float
    weight: float = 1.0


@dataclass
class PoseGraphOptimizationResult:
    poses: Dict[int, gtsam.Pose3]
    pose_matrices: Dict[int, np.ndarray]
    chunk_scales: Dict[int, float]
    initial_error: float
    final_error: float
    graph: gtsam.NonlinearFactorGraph
    initial_values: gtsam.Values
    optimized_values: gtsam.Values


class DA3ChunkPoseGraphOptimizer:
    def __init__(
        self,
        input_pose_is_w2c: bool = False,
        relative_rotation_sigma: float = 0.03,
        relative_translation_sigma: float = 0.05,
        prior_rotation_sigma: float = 0.01,
        prior_translation_sigma: float = 0.01,
        anchor_pose_sigma: float = 1e-6,
        anchor_log_scale_sigma: float = 1e-6,
        numeric_jacobian_eps: float = 1e-6,
        max_iterations: int = 100,
    ):
        """
        Pose graph optimizer for DA3 chunk poses.

        Assumption:
            Input poses are camera-to-world poses by default.

            If your poses are world-to-camera, set:
                input_pose_is_w2c=True

        Variables:
            X_i:
                Global image pose.

            s_c:
                Chunk scale. Internally optimized as log-scale:
                    l_c = log(s_c)

        Factors:
            1. Sequential relative pose factors from each DA3 chunk.
            2. Optional pose priors.
            3. Optional first-image identity prior.
            4. Optional first-chunk scale=1 prior.

        Notes:
            If you do not fix either scale or provide metric pose priors,
            global scale may be underconstrained.
        """

        self.input_pose_is_w2c = input_pose_is_w2c

        self.relative_rotation_sigma = relative_rotation_sigma
        self.relative_translation_sigma = relative_translation_sigma

        self.prior_rotation_sigma = prior_rotation_sigma
        self.prior_translation_sigma = prior_translation_sigma

        self.anchor_pose_sigma = anchor_pose_sigma
        self.anchor_log_scale_sigma = anchor_log_scale_sigma

        self.numeric_jacobian_eps = numeric_jacobian_eps
        self.max_iterations = max_iterations

    def optimize(
        self,
        chunks: Sequence[PoseChunk | Dict[str, Any]],
        pose_priors: Optional[Sequence[ImagePosePrior | Dict[str, Any]]] = None,
        scale_priors: Optional[Sequence[ChunkScalePrior | Dict[str, Any]]] = None,
        fix_first_pose_identity: bool = True,
        fix_first_chunk_scale_one: bool = True,
    ) -> PoseGraphOptimizationResult:
        """
        Optimize global image poses and per-chunk scales.

        Args:
            chunks:
                List of PoseChunk, or dictionaries like:

                    {
                        "chunk_id": 0,
                        "image_ids": [0, 1, 2, ...],
                        "poses": [T0, T1, T2, ...],
                        "scale": 1.0,
                        "weight": 1.0,
                    }

                Poses can be gtsam.Pose3 or 4x4 numpy arrays.

            pose_priors:
                List of ImagePosePrior, or dictionaries like:

                    {
                        "image_id": 0,
                        "pose": T_global,
                        "weight": 10.0,
                    }

            fix_first_pose_identity:
                If True, adds a strong prior:
                    first image pose = identity

            fix_first_chunk_scale_one:
                If True, adds a strong prior:
                    first chunk scale = 1

        Returns:
            PoseGraphOptimizationResult
        """

        chunks = [self._normalize_chunk(c, fallback_chunk_id=i) for i, c in enumerate(chunks)]
        pose_priors = [self._normalize_prior(p) for p in (pose_priors or [])]

        if len(chunks) == 0:
            raise ValueError("No chunks were provided.")

        graph = gtsam.NonlinearFactorGraph()
        initial = self._build_initial_values(chunks, pose_priors, fix_first_pose_identity)

        # Add DA3 chunk relative constraints.
        for chunk in chunks:
            scale_key = self._scale_key(chunk.chunk_id)

            noise = self._pose_noise(
                rotation_sigma=self.relative_rotation_sigma,
                translation_sigma=self.relative_translation_sigma,
                weight=chunk.weight,
            )

            local_poses = [self._as_pose3(p) for p in chunk.poses]

            for k in range(len(chunk.image_ids) - 1):
                image_id_i = int(chunk.image_ids[k])
                image_id_j = int(chunk.image_ids[k + 1])

                pose_i_local = local_poses[k]
                pose_j_local = local_poses[k + 1]

                # Sequential local relative pose.
                rel_ij_local = pose_i_local.between(pose_j_local)

                factor = self._make_scaled_between_factor(
                    image_id_i=image_id_i,
                    image_id_j=image_id_j,
                    chunk_id=chunk.chunk_id,
                    rel_ij_unscaled=rel_ij_local,
                    noise_model=noise,
                )

                graph.add(factor)

        # Add user-provided global pose priors.
        for prior in pose_priors:
            pose_key = self._pose_key(prior.image_id)
            prior_pose = self._as_pose3(prior.pose)

            prior_noise = self._pose_noise(
                rotation_sigma=self.prior_rotation_sigma,
                translation_sigma=self.prior_translation_sigma,
                weight=prior.weight,
            )

            graph.add(gtsam.PriorFactorPose3(pose_key, prior_pose, prior_noise))

        for prior in scale_priors:
            chunk_id = int(prior["chunk_id"]) if isinstance(prior, dict) else int(prior.chunk_id)
            scale = float(prior["scale"]) if isinstance(prior, dict) else float(prior.scale)
            weight = float(prior.get("weight", 1.0)) if isinstance(prior, dict) else float(prior.weight)

            if scale <= 0 or not np.isfinite(scale):
                raise ValueError(f"Invalid scale prior for chunk {chunk_id}: {scale}")

            sigma = 1.0 / math.sqrt(weight)

            graph.add(
                self._make_log_scale_prior_factor(
                    scale_key=self._scale_key(chunk_id),
                    target_log_scale=math.log(scale),
                    sigma=sigma,
                )
            )

        # Strongly anchor the first image pose to identity if requested.
        if fix_first_pose_identity:
            first_image_id = int(chunks[0].image_ids[0])
            pose_key = self._pose_key(first_image_id)

            anchor_noise = gtsam.noiseModel.Isotropic.Sigma(6, self.anchor_pose_sigma)
            graph.add(gtsam.PriorFactorPose3(pose_key, gtsam.Pose3(), anchor_noise))

        # Strongly anchor the first chunk scale to 1 if requested.
        if fix_first_chunk_scale_one:
            first_chunk_id = chunks[0].chunk_id
            scale_key = self._scale_key(first_chunk_id)

            graph.add(
                self._make_log_scale_prior_factor(
                    scale_key=scale_key,
                    target_log_scale=0.0,
                    sigma=self.anchor_log_scale_sigma,
                )
            )

        initial_error = graph.error(initial)

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iterations)

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        optimized = optimizer.optimize()

        final_error = graph.error(optimized)

        all_image_ids = self._collect_image_ids(chunks, pose_priors)
        all_chunk_ids = [c.chunk_id for c in chunks]

        optimized_poses = {image_id: optimized.atPose3(self._pose_key(image_id)) for image_id in sorted(all_image_ids)}

        optimized_pose_matrices = {image_id: self.pose3_to_matrix(pose) for image_id, pose in optimized_poses.items()}

        optimized_scales = {
            chunk_id: float(np.exp(optimized.atVector(self._scale_key(chunk_id))[0])) for chunk_id in all_chunk_ids
        }

        return PoseGraphOptimizationResult(
            poses=optimized_poses,
            pose_matrices=optimized_pose_matrices,
            chunk_scales=optimized_scales,
            initial_error=float(initial_error),
            final_error=float(final_error),
            graph=graph,
            initial_values=initial,
            optimized_values=optimized,
        )

    # -------------------------------------------------------------------------
    # Factor construction
    # -------------------------------------------------------------------------

    def _make_scaled_between_factor(
        self,
        image_id_i: int,
        image_id_j: int,
        chunk_id: int,
        rel_ij_unscaled: gtsam.Pose3,
        noise_model: gtsam.noiseModel.Base,
    ) -> gtsam.CustomFactor:
        """
        Custom factor:

            X_i.between(X_j) ~= Pose3(R_local_ij, exp(log_scale_c) * t_local_ij)

        Error:

            Logmap( measured_scaled.inverse() * predicted )
        """

        pose_key_i = self._pose_key(image_id_i)
        pose_key_j = self._pose_key(image_id_j)
        scale_key = self._scale_key(chunk_id)

        keys = [pose_key_i, pose_key_j, scale_key]

        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, jacobians):
            def err_at(v: gtsam.Values) -> np.ndarray:
                return self._scaled_between_error(
                    values=v,
                    pose_key_i=pose_key_i,
                    pose_key_j=pose_key_j,
                    scale_key=scale_key,
                    rel_ij_unscaled=rel_ij_unscaled,
                )

            error = err_at(values)

            if jacobians is not None:
                jacobians[0] = np.asfortranarray(self._numerical_pose_jacobian(values, pose_key_i, err_at))
                jacobians[1] = np.asfortranarray(self._numerical_pose_jacobian(values, pose_key_j, err_at))
                jacobians[2] = np.asfortranarray(self._numerical_vector_jacobian(values, scale_key, err_at, dim=1))

            return error

        return gtsam.CustomFactor(noise_model, keys, error_func)

    def _scaled_between_error(
        self,
        values: gtsam.Values,
        pose_key_i: int,
        pose_key_j: int,
        scale_key: int,
        rel_ij_unscaled: gtsam.Pose3,
    ) -> np.ndarray:
        pose_i = values.atPose3(pose_key_i)
        pose_j = values.atPose3(pose_key_j)

        predicted_rel = pose_i.between(pose_j)

        log_scale = float(values.atVector(scale_key)[0])
        scale = math.exp(log_scale)

        measured_rel_scaled = self._scale_pose_translation(rel_ij_unscaled, scale)

        error_pose = measured_rel_scaled.inverse().compose(predicted_rel)
        error = gtsam.Pose3.Logmap(error_pose)

        return np.asarray(error, dtype=np.float64).reshape(6)

    def _make_log_scale_prior_factor(
        self,
        scale_key: int,
        target_log_scale: float,
        sigma: float,
    ) -> gtsam.CustomFactor:
        noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, jacobians):
            estimate = values.atVector(scale_key)
            error = np.array([estimate[0] - target_log_scale], dtype=np.float64)

            if jacobians is not None:
                jacobians[0] = np.asfortranarray(np.eye(1, dtype=np.float64))

            return error

        return gtsam.CustomFactor(noise, [scale_key], error_func)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _build_initial_values(
        self,
        chunks: Sequence[PoseChunk],
        pose_priors: Sequence[ImagePosePrior],
        fix_first_pose_identity: bool,
    ) -> gtsam.Values:
        values = gtsam.Values()

        # Insert chunk log-scales.
        for chunk in chunks:
            if chunk.scale <= 0:
                raise ValueError(f"Chunk {chunk.chunk_id} has non-positive scale: {chunk.scale}")

            values.insert(
                self._scale_key(chunk.chunk_id),
                np.array([math.log(float(chunk.scale))], dtype=np.float64),
            )

        # Build initial global poses.
        pose_init: Dict[int, gtsam.Pose3] = {}

        # Priors are good initial seeds.
        for prior in pose_priors:
            pose_init[int(prior.image_id)] = self._as_pose3(prior.pose)

        # Optionally force first pose initialization to identity.
        first_image_id = int(chunks[0].image_ids[0])
        if fix_first_pose_identity:
            pose_init[first_image_id] = gtsam.Pose3()

        # Propagate through chunks.
        # Repeated passes help when chunks overlap but are not ordered perfectly.
        for _ in range(max(2, len(chunks))):
            changed = False

            for chunk in chunks:
                image_ids = [int(x) for x in chunk.image_ids]
                local_poses = [self._as_pose3(p) for p in chunk.poses]
                scale = float(chunk.scale)

                if len(image_ids) == 0:
                    continue

                # If this chunk has no initialized pose, seed its first pose.
                if not any(image_id in pose_init for image_id in image_ids):
                    pose_init[image_ids[0]] = gtsam.Pose3()
                    changed = True

                # Forward propagation.
                for k in range(len(image_ids) - 1):
                    i = image_ids[k]
                    j = image_ids[k + 1]

                    rel_ij = local_poses[k].between(local_poses[k + 1])
                    rel_ij_scaled = self._scale_pose_translation(rel_ij, scale)

                    if i in pose_init and j not in pose_init:
                        pose_init[j] = pose_init[i].compose(rel_ij_scaled)
                        changed = True

                # Backward propagation.
                for k in reversed(range(len(image_ids) - 1)):
                    i = image_ids[k]
                    j = image_ids[k + 1]

                    rel_ij = local_poses[k].between(local_poses[k + 1])
                    rel_ij_scaled = self._scale_pose_translation(rel_ij, scale)

                    if j in pose_init and i not in pose_init:
                        pose_init[i] = pose_init[j].compose(rel_ij_scaled.inverse())
                        changed = True

            if not changed:
                break

        # Insert all image pose variables.
        all_image_ids = self._collect_image_ids(chunks, pose_priors)

        for image_id in sorted(all_image_ids):
            if image_id not in pose_init:
                pose_init[image_id] = gtsam.Pose3()

            values.insert(self._pose_key(image_id), pose_init[image_id])

        return values

    # -------------------------------------------------------------------------
    # Numerical Jacobians
    # -------------------------------------------------------------------------

    def _numerical_pose_jacobian(
        self,
        values: gtsam.Values,
        key: int,
        error_func,
    ) -> np.ndarray:
        eps = self.numeric_jacobian_eps
        e0 = error_func(values)
        jac = np.zeros((e0.shape[0], 6), dtype=np.float64)

        pose = values.atPose3(key)

        for d in range(6):
            delta = np.zeros(6, dtype=np.float64)
            delta[d] = eps

            values_plus = gtsam.Values(values)
            values_minus = gtsam.Values(values)

            pose_plus = pose.compose(gtsam.Pose3.Expmap(delta))
            pose_minus = pose.compose(gtsam.Pose3.Expmap(-delta))

            values_plus.update(key, pose_plus)
            values_minus.update(key, pose_minus)

            e_plus = error_func(values_plus)
            e_minus = error_func(values_minus)

            jac[:, d] = (e_plus - e_minus) / (2.0 * eps)

        return jac

    def _numerical_vector_jacobian(
        self,
        values: gtsam.Values,
        key: int,
        error_func,
        dim: int,
    ) -> np.ndarray:
        eps = self.numeric_jacobian_eps
        e0 = error_func(values)
        jac = np.zeros((e0.shape[0], dim), dtype=np.float64)

        x = np.asarray(values.atVector(key), dtype=np.float64).reshape(dim)

        for d in range(dim):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[d] += eps
            x_minus[d] -= eps

            values_plus = gtsam.Values(values)
            values_minus = gtsam.Values(values)

            values_plus.update(key, x_plus)
            values_minus.update(key, x_minus)

            e_plus = error_func(values_plus)
            e_minus = error_func(values_minus)

            jac[:, d] = (e_plus - e_minus) / (2.0 * eps)

        return jac

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def _pose_key(self, image_id: int) -> int:
        return gtsam.symbol("x", int(image_id))

    def _scale_key(self, chunk_id: int) -> int:
        return gtsam.symbol("s", int(chunk_id))

    def _as_pose3(self, pose: Any) -> gtsam.Pose3:
        if isinstance(pose, gtsam.Pose3):
            p = pose
        else:
            T = np.asarray(pose, dtype=np.float64)

            if T.shape == (3, 4):
                T4 = np.eye(4, dtype=np.float64)
                T4[:3, :] = T
                T = T4

            if T.shape != (4, 4):
                raise ValueError(f"Expected pose shape (4, 4) or (3, 4), got {T.shape}")

            R = T[:3, :3]
            t = T[:3, 3]

            p = gtsam.Pose3(
                gtsam.Rot3(R),
                gtsam.Point3(float(t[0]), float(t[1]), float(t[2])),
            )

        if self.input_pose_is_w2c:
            p = p.inverse()

        return p

    @staticmethod
    def pose3_to_matrix(pose: gtsam.Pose3) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = pose.rotation().matrix()
        T[:3, 3] = np.asarray(pose.translation()).reshape(3)
        return T

    @staticmethod
    def _scale_pose_translation(pose: gtsam.Pose3, scale: float) -> gtsam.Pose3:
        t = np.asarray(pose.translation()).reshape(3)
        t_scaled = scale * t

        return gtsam.Pose3(
            pose.rotation(),
            gtsam.Point3(
                float(t_scaled[0]),
                float(t_scaled[1]),
                float(t_scaled[2]),
            ),
        )

    def _pose_noise(
        self,
        rotation_sigma: float,
        translation_sigma: float,
        weight: float,
    ) -> gtsam.noiseModel.Base:
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")

        # weight is interpreted as information multiplier:
        #     larger weight -> smaller sigma -> stronger factor
        scale = 1.0 / math.sqrt(weight)

        sigmas = np.array(
            [
                rotation_sigma * scale,
                rotation_sigma * scale,
                rotation_sigma * scale,
                translation_sigma * scale,
                translation_sigma * scale,
                translation_sigma * scale,
            ],
            dtype=np.float64,
        )

        return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    def _normalize_chunk(
        self,
        chunk: PoseChunk | Dict[str, Any],
        fallback_chunk_id: int,
    ) -> PoseChunk:
        if isinstance(chunk, PoseChunk):
            normalized = chunk
        elif isinstance(chunk, dict):
            chunk_id = int(chunk.get("chunk_id", fallback_chunk_id))

            poses = chunk["poses"]

            if isinstance(poses, dict):
                image_ids = chunk.get("image_ids", list(poses.keys()))
                pose_list = [poses[i] for i in image_ids]
            else:
                image_ids = chunk["image_ids"]
                pose_list = poses

            normalized = PoseChunk(
                chunk_id=chunk_id,
                image_ids=list(image_ids),
                poses=list(pose_list),
                scale=float(chunk.get("scale", chunk.get("scale_factor", 1.0))),
                weight=float(chunk.get("weight", 1.0)),
            )
        else:
            raise TypeError(f"Unsupported chunk type: {type(chunk)}")

        if len(normalized.image_ids) != len(normalized.poses):
            raise ValueError(f"Chunk {normalized.chunk_id}: image_ids and poses must have same length.")

        if len(normalized.image_ids) < 2:
            raise ValueError(f"Chunk {normalized.chunk_id}: at least 2 poses are required.")

        return normalized

    def _normalize_prior(
        self,
        prior: ImagePosePrior | Dict[str, Any],
    ) -> ImagePosePrior:
        if isinstance(prior, ImagePosePrior):
            return prior

        if isinstance(prior, dict):
            return ImagePosePrior(
                image_id=int(prior["image_id"]),
                pose=prior["pose"],
                weight=float(prior.get("weight", 1.0)),
            )

        raise TypeError(f"Unsupported prior type: {type(prior)}")

    @staticmethod
    def _collect_image_ids(
        chunks: Sequence[PoseChunk],
        priors: Sequence[ImagePosePrior],
    ) -> set[int]:
        image_ids = set()

        for chunk in chunks:
            for image_id in chunk.image_ids:
                image_ids.add(int(image_id))

        for prior in priors:
            image_ids.add(int(prior.image_id))

        return image_ids


if __name__ == "__main__":
    configure_logging()

    optimizer = DA3ChunkPoseGraphOptimizer(
        input_pose_is_w2c=False,  # set True if DA3 poses are world-to-camera
        relative_rotation_sigma=0.03,
        relative_translation_sigma=0.05,
    )

    chunks = [
        {
            "chunk_id": 0,
            "image_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "poses": poses_chunk_0,  # list of 4x4 c2w matrices or gtsam.Pose3
            "scale": 1.0,
            "weight": 1.0,
        },
        {
            "chunk_id": 1,
            "image_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "poses": poses_chunk_1,
            "scale": 1.0,
            "weight": 1.0,
        },
    ]

    pose_priors = [
        {
            "image_id": 0,
            "pose": np.eye(4),
            "weight": 100.0,
        }
    ]

    result = optimizer.optimize(
        chunks=chunks,
        pose_priors=pose_priors,
        fix_first_pose_identity=True,
        fix_first_chunk_scale_one=True,
    )

    logger.info("initial error: %s", result.initial_error)
    logger.info("final error: %s", result.final_error)

    optimized_poses = result.pose_matrices
    optimized_scales = result.chunk_scales

    for image_id, T in optimized_poses.items():
        logger.info("image %s\n%s", image_id, T)

    logger.info("chunk scales: %s", optimized_scales)
