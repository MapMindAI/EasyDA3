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


def intrinsics_to_K(intrinsics: np.ndarray) -> np.ndarray:
    """
    Accept either:
      - K: shape (3, 3)
      - [fx, fy, cx, cy]&#58; shape (4,)
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float64)

    if intrinsics.shape == (3, 3):
        return intrinsics

    if intrinsics.shape == (4,):
        fx, fy, cx, cy = intrinsics
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    raise ValueError("intrinsics must be shape (3, 3) or (4,) as [fx, fy, cx, cy]")


def sample_depth_bilinear(depth: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Bilinear depth sampling at floating-point pixel locations.
    Invalid/out-of-bound samples are returned as NaN.
    """
    h, w = depth.shape[:2]

    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)

    z = np.full(xs.shape, np.nan, dtype=np.float64)

    if not np.any(valid):
        return z

    xv = xs[valid]
    yv = ys[valid]

    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]

    wa = (x1v - xv) * (y1v - yv)
    wb = (x1v - xv) * (yv - y0v)
    wc = (xv - x0v) * (y1v - yv)
    wd = (xv - x0v) * (yv - y0v)

    Ia = depth[y0v, x0v].astype(np.float64)
    Ib = depth[y1v, x0v].astype(np.float64)
    Ic = depth[y0v, x1v].astype(np.float64)
    Id = depth[y1v, x1v].astype(np.float64)

    z[valid] = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return z


def backproject_pixels_to_camera(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Back-project pixels into camera coordinates.

    Pixel:
        u = x
        v = y

    Camera:
        X = (u - cx) / fx * Z
        Y = (v - cy) / fy * Z
        Z = depth
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X = (xs - cx) / fx * zs
    Y = (ys - cy) / fy * zs
    Z = zs

    return np.stack([X, Y, Z], axis=1).astype(np.float64)


def transform_points_c2w(points_cam: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    """
    Transform 3D points from camera coordinates to world coordinates.

    pose_c2w:
        shape (4, 4)
        maps camera coordinates to world coordinates
    """
    R = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]

    return points_cam @ R.T + t[None, :]


def transform_points_w2c(points_world: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    """
    Transform 3D points from world coordinates to camera coordinates.

    pose_c2w:
        shape (4, 4)
        maps camera coordinates to world coordinates
    """
    pose_w2c = np.linalg.inv(as_4x4(pose_c2w))
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    return points_world @ R.T + t[None, :]


def project_depth_to_frame(
    depth_src: np.ndarray,
    intrinsics_src: np.ndarray,
    pose_c2w_src: np.ndarray,
    intrinsics_dst: np.ndarray,
    pose_c2w_dst: np.ndarray,
    dst_shape,
    min_depth: float = 1e-6,
    max_depth: float | None = None,
    splat_radius: int = 1,
) -> np.ndarray:
    """
    Forward-project a source keyframe depth map into a destination camera.

    The result is a z-buffered destination depth map in destination camera
    coordinates. Pixels without projected support are NaN.
    """

    depth_src = np.asarray(depth_src, dtype=np.float32)
    if depth_src.ndim != 2:
        raise ValueError("depth_src must be a single-channel depth map with shape (H, W)")

    if len(dst_shape) < 2:
        raise ValueError("dst_shape must contain at least height and width")

    dst_h = int(dst_shape[0])
    dst_w = int(dst_shape[1])
    if dst_h <= 0 or dst_w <= 0:
        raise ValueError(f"invalid destination shape: {dst_shape}")

    K_src = intrinsics_to_K(intrinsics_src)
    K_dst = intrinsics_to_K(intrinsics_dst)
    pose_c2w_src = as_4x4(pose_c2w_src)
    pose_c2w_dst = as_4x4(pose_c2w_dst)

    valid = np.isfinite(depth_src) & (depth_src > min_depth)
    if max_depth is not None:
        valid &= depth_src < max_depth

    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    zs = depth_src[ys, xs].astype(np.float64)
    pts_src_cam = backproject_pixels_to_camera(xs.astype(np.float64), ys.astype(np.float64), zs, K_src)
    pts_world = transform_points_c2w(pts_src_cam, pose_c2w_src)
    pts_dst_cam = transform_points_w2c(pts_world, pose_c2w_dst)

    z_dst = pts_dst_cam[:, 2]
    valid_dst = np.isfinite(z_dst) & (z_dst > min_depth)
    if max_depth is not None:
        valid_dst &= z_dst < max_depth

    pts_dst_cam = pts_dst_cam[valid_dst]
    z_dst = z_dst[valid_dst]
    if len(z_dst) == 0:
        return np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    u = K_dst[0, 0] * (pts_dst_cam[:, 0] / z_dst) + K_dst[0, 2]
    v = K_dst[1, 1] * (pts_dst_cam[:, 1] / z_dst) + K_dst[1, 2]

    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    inside = (px >= 0) & (px < dst_w) & (py >= 0) & (py < dst_h)

    if not np.any(inside):
        return np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    px = px[inside]
    py = py[inside]
    z_dst = z_dst[inside].astype(np.float32)

    z_buffer = np.full((dst_h, dst_w), np.inf, dtype=np.float32)
    radius = max(0, int(splat_radius))

    for dy in range(-radius, radius + 1):
        yy = py + dy
        y_ok = (yy >= 0) & (yy < dst_h)

        if not np.any(y_ok):
            continue

        for dx in range(-radius, radius + 1):
            xx = px + dx
            ok = y_ok & (xx >= 0) & (xx < dst_w)

            if not np.any(ok):
                continue

            np.minimum.at(z_buffer, (yy[ok], xx[ok]), z_dst[ok])

    z_buffer[~np.isfinite(z_buffer)] = np.nan
    return z_buffer


def invert_pose_w2c_to_c2w(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    """
    OpenCV solvePnP returns world-to-camera:

        X_cur = R_w2c @ X_world + t_w2c

    We invert it to get camera-to-world:

        X_world = R_c2w @ X_cur + t_c2w
    """
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c.reshape(3)

    pose_c2w = np.eye(4, dtype=np.float64)
    pose_c2w[:3, :3] = R_c2w
    pose_c2w[:3, 3] = t_c2w

    return pose_c2w


def estimate_pose_c2w_cur(
    tracks: np.ndarray,
    depth_kf: np.ndarray,
    intrinsics_kf: np.ndarray,
    pose_c2w_kf: np.ndarray,
    intrinsics_cur: np.ndarray | None = None,
    min_depth: float = 1e-6,
    max_depth: float | None = None,
    reprojection_error_px: float = 4.0,
    confidence: float = 0.999,
    iterations_count: int = 200,
    min_inliers: int = 12,
    dist_coeffs: np.ndarray | None = None,
    return_info: bool = False,
):
    """
    Estimate current camera pose from 2D tracks and keyframe depth.

    Inputs
    ------
    tracks:
        np.ndarray, shape (N, 4)
        columns: [x_kf, y_kf, x_cur, y_cur]

    depth_kf:
        np.ndarray, shape (H, W)
        depth map of the keyframe, in keyframe camera coordinates.

    intrinsics_kf:
        np.ndarray
        Either K matrix shape (3, 3), or [fx, fy, cx, cy].

    pose_c2w_kf:
        np.ndarray, shape (4, 4)
        keyframe camera-to-world pose.

    intrinsics_cur:
        np.ndarray or None
        Current frame intrinsics. If None, uses intrinsics_kf.
        This is valid when keyframe/current come from the same camera
        and same image resolution.

    Output
    ------
    pose_c2w_cur:
        np.ndarray, shape (4, 4)
        Current camera-to-world pose.

    If return_info=True, returns:
        pose_c2w_cur, info
    """

    tracks = np.asarray(tracks, dtype=np.float64)
    depth_kf = np.asarray(depth_kf)
    pose_c2w_kf = np.asarray(pose_c2w_kf, dtype=np.float64)

    if tracks.ndim != 2 or tracks.shape[1] != 4:
        raise ValueError("tracks must have shape (N, 4), columns [x_kf, y_kf, x_cur, y_cur]")

    if depth_kf.ndim != 2:
        raise ValueError("depth_kf must be a single-channel depth map with shape (H, W)")

    if pose_c2w_kf.shape != (4, 4):
        raise ValueError("pose_c2w_kf must have shape (4, 4)")

    K_kf = intrinsics_to_K(intrinsics_kf)
    K_cur = intrinsics_to_K(intrinsics_cur) if intrinsics_cur is not None else K_kf.copy()

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    x_kf = tracks[:, 0]
    y_kf = tracks[:, 1]
    x_cur = tracks[:, 2]
    y_cur = tracks[:, 3]

    # 1. Sample keyframe depth
    z_kf = sample_depth_bilinear(depth_kf, x_kf, y_kf)

    # 2. Filter invalid depth and invalid tracks
    valid = np.isfinite(z_kf) & np.isfinite(x_cur) & np.isfinite(y_cur)
    valid &= z_kf > min_depth

    if max_depth is not None:
        valid &= z_kf < max_depth

    x_kf_valid = x_kf[valid]
    y_kf_valid = y_kf[valid]
    z_kf_valid = z_kf[valid]

    x_cur_valid = x_cur[valid]
    y_cur_valid = y_cur[valid]

    if len(z_kf_valid) < 6:
        raise RuntimeError(f"Not enough valid 2D-3D correspondences: {len(z_kf_valid)}")

    # 3. Backproject keyframe pixels to keyframe camera 3D
    pts_kf_cam = backproject_pixels_to_camera(
        x_kf_valid,
        y_kf_valid,
        z_kf_valid,
        K_kf,
    )

    # 4. Transform keyframe camera 3D to world 3D
    # pts_world = transform_points_c2w(pts_kf_cam, pose_c2w_kf)

    # 5. Current frame 2D observations
    pts_cur_2d = np.stack([x_cur_valid, y_cur_valid], axis=1).astype(np.float64)

    # OpenCV expects contiguous arrays
    object_points = np.ascontiguousarray(pts_kf_cam.reshape(-1, 1, 3))
    image_points = np.ascontiguousarray(pts_cur_2d.reshape(-1, 1, 2))

    # 6. PnP RANSAC: world -> current camera
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=K_cur,
        distCoeffs=dist_coeffs,
        iterationsCount=iterations_count,
        reprojectionError=reprojection_error_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not ok or inliers is None:
        raise RuntimeError("solvePnPRansac failed")

    inliers = inliers.reshape(-1)

    if len(inliers) < min_inliers:
        raise RuntimeError(f"Too few PnP inliers: {len(inliers)} / {len(object_points)}")

    # 7. Optional refinement using only RANSAC inliers
    inlier_object_points = object_points[inliers]
    inlier_image_points = image_points[inliers]

    ok_refine, rvec, tvec = cv2.solvePnP(
        objectPoints=inlier_object_points,
        imagePoints=inlier_image_points,
        cameraMatrix=K_cur,
        distCoeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok_refine:
        raise RuntimeError("solvePnP refinement failed")

    R_w2c, _ = cv2.Rodrigues(rvec)
    t_w2c = tvec.reshape(3)

    # 8. Convert world-to-current-camera to current-camera-to-world
    pose_c2kf = invert_pose_w2c_to_c2w(R_w2c, t_w2c)
    pose_c2w_cur = np.dot(pose_c2w_kf, pose_c2kf)

    if not return_info:
        return pose_c2w_cur

    info = {
        "num_tracks": int(len(tracks)),
        "num_valid_depth": int(len(z_kf_valid)),
        "num_inliers": int(len(inliers)),
        "inlier_ratio": float(len(inliers) / max(1, len(z_kf_valid))),
        "valid_mask_original_tracks": valid,
        "pnp_inlier_indices_after_depth_filter": inliers,
        "pose_w2c_cur_R": R_w2c,
        "pose_w2c_cur_t": t_w2c,
    }

    return pose_c2w_cur, info
