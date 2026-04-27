import math

import numpy as np
import pytest
import gtsam
import types

from da3_pose_graph_optimizer import (
    DA3ChunkPoseGraphOptimizer,
    PoseChunk,
    ImagePosePrior,
)


def make_pose_xyz(x: float, y: float = 0.0, z: float = 0.0) -> gtsam.Pose3:
    """
    Create a simple camera-to-world pose with identity rotation.
    """
    return gtsam.Pose3(
        gtsam.Rot3(),
        gtsam.Point3(float(x), float(y), float(z)),
    )


def pose_translation(pose: gtsam.Pose3) -> np.ndarray:
    return np.asarray(pose.translation(), dtype=np.float64).reshape(3)


def assert_pose_translation_close(
    pose: gtsam.Pose3,
    expected_xyz,
    atol: float = 1e-3,
):
    actual = pose_translation(pose)
    expected = np.asarray(expected_xyz, dtype=np.float64).reshape(3)

    np.testing.assert_allclose(actual, expected, atol=atol)


def random_pose_near(
    base_pose: gtsam.Pose3,
    rng: np.random.Generator,
    rotation_sigma: float = 0.10,
    translation_sigma: float = 1.0,
) -> gtsam.Pose3:
    """
    Create a random noisy pose near base_pose.

    The noise is applied in the local tangent space of SE(3).
    """
    delta = np.zeros(6, dtype=np.float64)

    # GTSAM Pose3 tangent vector convention:
    # first 3 are rotation, last 3 are translation.
    delta[0:3] = rng.normal(0.0, rotation_sigma, size=3)
    delta[3:6] = rng.normal(0.0, translation_sigma, size=3)

    return base_pose.compose(gtsam.Pose3.Expmap(delta))


def assert_scale_close(
    actual: float,
    expected: float,
    atol: float = 1e-3,
):
    assert abs(actual - expected) < atol, f"scale {actual} != expected {expected}"


def test_single_chunk_identity_scale_recovers_line_trajectory():
    """
    Simple sanity test.

    One chunk:
        image ids: 0..9
        local poses are already metric
        scale = 1

    Expected:
        optimized pose i should be at x = i
        optimized scale should remain 1
    """

    image_ids = list(range(10))
    local_poses = [make_pose_xyz(float(i)) for i in image_ids]

    chunks = [
        PoseChunk(
            chunk_id=0,
            image_ids=image_ids,
            poses=local_poses,
            scale=1.0,
            weight=1.0,
        )
    ]

    optimizer = DA3ChunkPoseGraphOptimizer(
        input_pose_is_w2c=False,
        relative_rotation_sigma=0.01,
        relative_translation_sigma=0.01,
        max_iterations=50,
    )

    result = optimizer.optimize(
        chunks=chunks,
        pose_priors=[],
        fix_first_pose_identity=True,
        fix_first_chunk_scale_one=True,
    )

    assert result.final_error < 1e-8

    for image_id in image_ids:
        assert_pose_translation_close(
            result.poses[image_id],
            expected_xyz=[float(image_id), 0.0, 0.0],
            atol=1e-5,
        )

    assert_scale_close(result.chunk_scales[0], 1.0, atol=1e-6)


def test_two_overlapping_chunks_recover_second_chunk_scale():
    """
    Main DA3-style test.

    Ground-truth global trajectory:
        image i is at x = i

    Chunk 0:
        image ids: 0..9
        local spacing = 1
        scale = 1

    Chunk 1:
        image ids: 5..14
        local spacing = 2

    Since chunk 1 local translation is 2x too large, the optimizer should recover:

        scale_1 ~= 0.5

    The overlap images 5..9 connect the two chunks.
    """

    chunk0_image_ids = list(range(0, 10))
    chunk1_image_ids = list(range(5, 15))

    # Chunk 0 is already metric.
    chunk0_local_poses = [make_pose_xyz(float(i)) for i in range(len(chunk0_image_ids))]

    # Chunk 1 is local to the chunk and has 2x translation spacing.
    # It starts at local x=0 for image 5.
    chunk1_local_poses = [make_pose_xyz(float(2 * i)) for i in range(len(chunk1_image_ids))]

    chunks = [
        PoseChunk(
            chunk_id=0,
            image_ids=chunk0_image_ids,
            poses=chunk0_local_poses,
            scale=1.0,
            weight=1.0,
        ),
        PoseChunk(
            chunk_id=1,
            image_ids=chunk1_image_ids,
            poses=chunk1_local_poses,
            scale=1.0,  # intentionally wrong initial scale
            weight=1.0,
        ),
    ]

    optimizer = DA3ChunkPoseGraphOptimizer(
        input_pose_is_w2c=False,
        relative_rotation_sigma=0.01,
        relative_translation_sigma=0.05,
        anchor_pose_sigma=1e-8,
        anchor_log_scale_sigma=1e-8,
        max_iterations=100,
    )

    result = optimizer.optimize(
        chunks=chunks,
        pose_priors=[],
        fix_first_pose_identity=True,
        fix_first_chunk_scale_one=True,
    )

    assert result.final_error < result.initial_error

    # Chunk 0 should remain metric.
    assert_scale_close(result.chunk_scales[0], 1.0, atol=1e-5)

    # Chunk 1 local translation is 2x too large, so scale should be 0.5.
    assert_scale_close(result.chunk_scales[1], 0.5, atol=1e-3)

    # All global poses should lie on x = image_id.
    for image_id in range(15):
        assert_pose_translation_close(
            result.poses[image_id],
            expected_xyz=[float(image_id), 0.0, 0.0],
            atol=1e-3,
        )


def test_pose_prior_can_anchor_pose_without_identity_flag():
    """
    Test user-provided pose prior.

    We disable fix_first_pose_identity and instead provide a prior:

        image 0 pose = x = 10

    Then the whole trajectory should be shifted by +10.
    """

    image_ids = list(range(10))
    local_poses = [make_pose_xyz(float(i)) for i in image_ids]

    chunks = [
        PoseChunk(
            chunk_id=0,
            image_ids=image_ids,
            poses=local_poses,
            scale=1.0,
            weight=1.0,
        )
    ]

    pose_priors = [
        ImagePosePrior(
            image_id=0,
            pose=make_pose_xyz(10.0),
            weight=1000.0,
        )
    ]

    optimizer = DA3ChunkPoseGraphOptimizer(
        input_pose_is_w2c=False,
        relative_rotation_sigma=0.01,
        relative_translation_sigma=0.01,
        prior_rotation_sigma=1e-4,
        prior_translation_sigma=1e-4,
        max_iterations=50,
    )

    result = optimizer.optimize(
        chunks=chunks,
        pose_priors=pose_priors,
        fix_first_pose_identity=False,
        fix_first_chunk_scale_one=True,
    )

    assert result.final_error < 1e-6

    for image_id in image_ids:
        expected_x = 10.0 + float(image_id)

        assert_pose_translation_close(
            result.poses[image_id],
            expected_xyz=[expected_x, 0.0, 0.0],
            atol=1e-4,
        )

    assert_scale_close(result.chunk_scales[0], 1.0, atol=1e-6)


def test_optimizer_accepts_numpy_4x4_poses():
    """
    Test that numpy 4x4 poses are accepted, not only gtsam.Pose3.
    """

    def pose_to_matrix(pose: gtsam.Pose3) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = pose.rotation().matrix()
        T[:3, 3] = pose_translation(pose)
        return T

    image_ids = list(range(5))

    local_poses_gtsam = [make_pose_xyz(float(i)) for i in image_ids]
    local_poses_np = [pose_to_matrix(p) for p in local_poses_gtsam]

    chunks = [
        {
            "chunk_id": 0,
            "image_ids": image_ids,
            "poses": local_poses_np,
            "scale": 1.0,
            "weight": 1.0,
        }
    ]

    optimizer = DA3ChunkPoseGraphOptimizer(
        input_pose_is_w2c=False,
        max_iterations=50,
    )

    result = optimizer.optimize(
        chunks=chunks,
        pose_priors=[],
        fix_first_pose_identity=True,
        fix_first_chunk_scale_one=True,
    )

    for image_id in image_ids:
        assert_pose_translation_close(
            result.poses[image_id],
            expected_xyz=[float(image_id), 0.0, 0.0],
            atol=1e-5,
        )


def test_two_overlapping_chunks_with_random_initial_poses():
    """
    DA3-style test with random initial global poses.

    Ground truth:
        image i is at x = i

    Chunk 0:
        image ids 0..9
        local spacing = 1
        scale should be 1

    Chunk 1:
        image ids 5..14
        local spacing = 2
        scale should be 0.5

    The optimizer is deliberately initialized with random noisy poses.
    """

    rng = np.random.default_rng(42)

    chunk0_image_ids = list(range(0, 10))
    chunk1_image_ids = list(range(5, 15))

    # Ground-truth global poses.
    gt_poses = {image_id: make_pose_xyz(float(image_id)) for image_id in range(15)}

    # Chunk 0 local poses are metric.
    chunk0_local_poses = [make_pose_xyz(float(i)) for i in range(len(chunk0_image_ids))]

    # Chunk 1 local poses have 2x too-large translation.
    # Therefore the optimizer should recover scale_1 ~= 0.5.
    chunk1_local_poses = [make_pose_xyz(float(2 * i)) for i in range(len(chunk1_image_ids))]

    chunks = [
        PoseChunk(
            chunk_id=0,
            image_ids=chunk0_image_ids,
            poses=chunk0_local_poses,
            scale=1.3,  # intentionally wrong initial scale
            weight=1.0,
        ),
        PoseChunk(
            chunk_id=1,
            image_ids=chunk1_image_ids,
            poses=chunk1_local_poses,
            scale=0.8,  # intentionally wrong initial scale
            weight=1.0,
        ),
    ]

    optimizer = DA3ChunkPoseGraphOptimizer(
        input_pose_is_w2c=False,
        relative_rotation_sigma=0.01,
        relative_translation_sigma=0.05,
        anchor_pose_sigma=1e-8,
        anchor_log_scale_sigma=1e-8,
        max_iterations=200,
    )

    def random_build_initial_values(
        self,
        chunks,
        pose_priors,
        fix_first_pose_identity,
    ) -> gtsam.Values:
        """
        Replacement for optimizer._build_initial_values() used only in this test.

        It inserts random noisy initial poses instead of the normal propagated poses.
        """
        values = gtsam.Values()

        # Random initial log-scales.
        for chunk in chunks:
            random_scale = float(rng.uniform(0.4, 1.6))
            values.insert(
                self._scale_key(chunk.chunk_id),
                np.array([math.log(random_scale)], dtype=np.float64),
            )

        all_image_ids = self._collect_image_ids(chunks, pose_priors)
        first_image_id = int(chunks[0].image_ids[0])

        for image_id in sorted(all_image_ids):
            if fix_first_pose_identity and image_id == first_image_id:
                initial_pose = gtsam.Pose3()
            else:
                initial_pose = random_pose_near(
                    gt_poses[image_id],
                    rng=rng,
                    rotation_sigma=0.10,
                    translation_sigma=1.0,
                )

            values.insert(self._pose_key(image_id), initial_pose)

        return values

    # Monkey-patch only this optimizer instance.
    optimizer._build_initial_values = types.MethodType(
        random_build_initial_values,
        optimizer,
    )

    result = optimizer.optimize(
        chunks=chunks,
        pose_priors=[],
        fix_first_pose_identity=True,
        fix_first_chunk_scale_one=True,
    )

    assert result.final_error < result.initial_error

    # First chunk is anchored to scale 1.
    assert abs(result.chunk_scales[0] - 1.0) < 1e-4

    # Second chunk local motion is 2x too large, so estimated scale should be 0.5.
    assert abs(result.chunk_scales[1] - 0.5) < 1e-3

    # Global poses should recover x = image_id.
    for image_id in range(15):
        assert_pose_translation_close(
            result.poses[image_id],
            expected_xyz=[float(image_id), 0.0, 0.0],
            atol=1e-3,
        )
