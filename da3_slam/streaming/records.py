from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class KeyframeRecord:
    image_id: int
    source_frame_id: int
    image_bgr: np.ndarray

    depth: Optional[np.ndarray] = None
    depth_conf: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None

    pose_c2w: Optional[np.ndarray] = None
    pose_w2c: Optional[np.ndarray] = None

    da3_local_w2c: Optional[np.ndarray] = None
    last_chunk_id: Optional[int] = None


@dataclass
class DA3ChunkRecord:
    chunk_id: int
    image_ids: List[int]

    depth_list: List[np.ndarray]
    depth_conf_list: List[np.ndarray]
    intrinsics_list: List[np.ndarray]
    extrinsics_list: List[np.ndarray]

    scale: float = 1.0
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

    estimated_pose_c2w: Optional[np.ndarray] = None
    estimated_pose_w2c: Optional[np.ndarray] = None
    pose_reference_keyframe_id: Optional[int] = None
    pose_reference_source_frame_id: Optional[int] = None
    pose_estimation_info: Optional[Any] = None
    pose_estimation_error: Optional[str] = None

    projected_depth_keyframe_id: Optional[int] = None
    projected_depth_source_keyframe_id: Optional[int] = None
    projected_depth_valid_pixels: int = 0
