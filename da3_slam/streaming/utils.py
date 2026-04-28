from __future__ import annotations

import cv2
import numpy as np


def as_4x4(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform)

    if transform.shape == (4, 4):
        return transform.astype(np.float64)

    if transform.shape == (3, 4):
        transform_44 = np.eye(4, dtype=np.float64)
        transform_44[:3, :4] = transform.astype(np.float64)
        return transform_44

    raise ValueError(f"Expected pose shape (3, 4) or (4, 4), got {transform.shape}")


def to_bgr_uint8(image: np.ndarray) -> np.ndarray:
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

        return img.copy()

    raise ValueError(f"Unsupported image shape: {img.shape}")


def depth_to_u8(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)

    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.uint8)

    values = depth[valid]
    depth_min = np.percentile(values, 2)
    depth_max = np.percentile(values, 98)

    depth_norm = np.clip(depth, depth_min, depth_max)
    depth_norm = (depth_norm - depth_min) / (depth_max - depth_min + 1e-8)
    depth_norm = 1.0 - depth_norm

    return (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
