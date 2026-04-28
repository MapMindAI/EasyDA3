from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .records import DA3ChunkRecord, KeyframeRecord
from .utils import depth_to_u8


class StreamingMapStorage:
    def __init__(self, output_dir: str | Path, save_depth_png_preview: bool = True):
        self.output_dir = Path(output_dir)
        self.keyframe_dir = self.output_dir / "keyframes"
        self.chunk_dir = self.output_dir / "chunks"
        self.save_depth_png_preview = bool(save_depth_png_preview)

        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

    def save_keyframe_rgb(self, keyframe: KeyframeRecord) -> None:
        path = self.keyframe_dir / f"kf_{keyframe.image_id:06d}_rgb.png"
        cv2.imwrite(str(path), keyframe.image_bgr)

    def save_keyframe(self, keyframe: KeyframeRecord) -> None:
        self.save_keyframe_rgb(keyframe)

        prefix = self.keyframe_dir / f"kf_{keyframe.image_id:06d}"

        if keyframe.depth is not None:
            np.save(str(prefix) + "_depth.npy", keyframe.depth.astype(np.float32))

            if self.save_depth_png_preview:
                depth_png = depth_to_u8(keyframe.depth)
                cv2.imwrite(str(prefix) + "_depth_vis.png", depth_png)

        if keyframe.intrinsics is not None:
            np.save(str(prefix) + "_intrinsics.npy", keyframe.intrinsics.astype(np.float64))

        if keyframe.pose_c2w is not None:
            np.save(str(prefix) + "_pose_c2w.npy", keyframe.pose_c2w.astype(np.float64))

        if keyframe.pose_w2c is not None:
            np.save(str(prefix) + "_pose_w2c.npy", keyframe.pose_w2c.astype(np.float64))

        meta = {
            "image_id": keyframe.image_id,
            "source_frame_id": keyframe.source_frame_id,
            "last_chunk_id": keyframe.last_chunk_id,
            "has_depth": keyframe.depth is not None,
            "has_intrinsics": keyframe.intrinsics is not None,
            "has_pose": keyframe.pose_c2w is not None,
            "rgb_path": f"kf_{keyframe.image_id:06d}_rgb.png",
            "depth_path": f"kf_{keyframe.image_id:06d}_depth.npy" if keyframe.depth is not None else None,
            "intrinsics_path": (
                f"kf_{keyframe.image_id:06d}_intrinsics.npy" if keyframe.intrinsics is not None else None
            ),
            "pose_c2w_path": f"kf_{keyframe.image_id:06d}_pose_c2w.npy" if keyframe.pose_c2w is not None else None,
            "pose_w2c_path": f"kf_{keyframe.image_id:06d}_pose_w2c.npy" if keyframe.pose_w2c is not None else None,
        }

        with open(str(prefix) + "_meta.json", "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

    def save_chunk(self, chunk: DA3ChunkRecord) -> None:
        chunk_path = self.chunk_dir / f"chunk_{chunk.chunk_id:06d}.json"

        meta = {
            "chunk_id": chunk.chunk_id,
            "image_ids": chunk.image_ids,
            "scale": chunk.scale,
            "initial_error": chunk.initial_error,
            "final_error": chunk.final_error,
        }

        with open(chunk_path, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

    def save_manifest(
        self,
        *,
        window_size: int,
        da3_stride_new_keyframes: int,
        optimizer_num_chunks: int,
        keyframes: Sequence[KeyframeRecord],
        chunks: Sequence[DA3ChunkRecord],
    ) -> None:
        manifest = {
            "window_size": window_size,
            "da3_stride_new_keyframes": da3_stride_new_keyframes,
            "optimizer_num_chunks": optimizer_num_chunks,
            "num_keyframes": len(keyframes),
            "num_chunks": len(chunks),
            "keyframes": [],
            "chunks": [],
        }

        for keyframe in keyframes:
            manifest["keyframes"].append(
                {
                    "image_id": keyframe.image_id,
                    "source_frame_id": keyframe.source_frame_id,
                    "last_chunk_id": keyframe.last_chunk_id,
                    "has_depth": keyframe.depth is not None,
                    "has_intrinsics": keyframe.intrinsics is not None,
                    "has_pose": keyframe.pose_c2w is not None,
                    "rgb_path": f"keyframes/kf_{keyframe.image_id:06d}_rgb.png",
                    "depth_path": (
                        f"keyframes/kf_{keyframe.image_id:06d}_depth.npy"
                        if keyframe.depth is not None
                        else None
                    ),
                    "intrinsics_path": (
                        f"keyframes/kf_{keyframe.image_id:06d}_intrinsics.npy"
                        if keyframe.intrinsics is not None
                        else None
                    ),
                    "pose_c2w_path": (
                        f"keyframes/kf_{keyframe.image_id:06d}_pose_c2w.npy"
                        if keyframe.pose_c2w is not None
                        else None
                    ),
                    "pose_w2c_path": (
                        f"keyframes/kf_{keyframe.image_id:06d}_pose_w2c.npy"
                        if keyframe.pose_w2c is not None
                        else None
                    ),
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

        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
