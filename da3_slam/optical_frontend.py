# Copyright 2026 MapMind Inc. All rights reserved.
#
# sparse-LK optical-flow frontend: it keeps a reference keyframe, tracks features from that keyframe through
# incoming frames, and promotes the current frame to a new keyframe once motion is large enough or tracking
# quality drops.
# * visual observation between keyframes will be saved.
# * new keyframe will be telled.

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class OpticalFlowResult:
    is_keyframe: bool
    tracks: np.ndarray  # shape: (N, 4), columns: x_kf, y_kf, x_cur, y_cur
    num_tracks: int
    median_pixel_motion: float
    mean_pixel_motion: float
    frame_id: int
    keyframe_id: int


class OpticalFlowKeyframeProcessor:
    def __init__(
        self,
        max_features: int = 1000,
        min_feature_distance: int = 8,
        quality_level: float = 0.01,
        block_size: int = 7,
        keyframe_pixel_threshold: float = 25.0,
        min_tracked_features: int = 150,
        lk_win_size=(21, 21),
        lk_max_level: int = 3,
        lk_max_iter: int = 30,
        lk_eps: float = 0.01,
        use_forward_backward_check: bool = True,
        forward_backward_threshold: float = 1.5,
    ):
        """
        Sparse optical flow processor for keyframe extraction.

        Args:
            max_features:
                Maximum number of features to detect on each keyframe.

            min_feature_distance:
                Minimum distance between detected features.

            quality_level:
                OpenCV goodFeaturesToTrack quality level.

            block_size:
                OpenCV goodFeaturesToTrack block size.

            keyframe_pixel_threshold:
                If median feature motion from the last keyframe to current frame
                is larger than this, the current frame becomes a new keyframe.

            min_tracked_features:
                If tracked feature count drops below this number, force a new keyframe.

            lk_win_size:
                Lucas-Kanade optical flow window size.

            lk_max_level:
                Pyramid levels for Lucas-Kanade optical flow.

            lk_max_iter:
                Maximum LK optimization iterations.

            lk_eps:
                LK optimization epsilon.

            use_forward_backward_check:
                If True, reject bad optical-flow tracks using forward-backward check.

            forward_backward_threshold:
                Maximum allowed forward-backward tracking error in pixels.
        """

        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=min_feature_distance,
            blockSize=block_size,
        )

        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                lk_max_iter,
                lk_eps,
            ),
        )

        self.keyframe_pixel_threshold = keyframe_pixel_threshold
        self.min_tracked_features = min_tracked_features
        self.use_forward_backward_check = use_forward_backward_check
        self.forward_backward_threshold = forward_backward_threshold

        self.initialized = False

        self.frame_id = -1
        self.keyframe_id = -1

        self.keyframe_gray = None
        self.last_gray = None

        # Feature coordinates in the keyframe.
        # Shape: (N, 2)
        self.keyframe_points = None

        # Current positions of those same keyframe features in the latest frame.
        # Shape: (N, 2)
        self.last_points = None

        self.image_shape = None

    def process(self, image: np.ndarray) -> OpticalFlowResult:
        """
        Process one image.

        Args:
            image:
                Input image. Can be BGR, RGB, or grayscale.
                The class only uses grayscale internally.

        Returns:
            OpticalFlowResult:
                is_keyframe:
                    Whether the current frame is selected as a new keyframe.

                tracks:
                    N x 4 array:
                    [x_keyframe, y_keyframe, x_current, y_current]
        """

        gray = self._to_gray(image)
        self.frame_id += 1

        if not self.initialized:
            self._initialize(gray)

            return OpticalFlowResult(
                is_keyframe=True,
                tracks=np.empty((0, 4), dtype=np.float32),
                num_tracks=0,
                median_pixel_motion=0.0,
                mean_pixel_motion=0.0,
                frame_id=self.frame_id,
                keyframe_id=self.keyframe_id,
            )

        if gray.shape != self.image_shape:
            raise ValueError(
                f"Image size changed from {self.image_shape} to {gray.shape}. "
                "This processor expects fixed image size."
            )

        if self.last_points is None or len(self.last_points) == 0:
            self._set_new_keyframe(gray)

            return OpticalFlowResult(
                is_keyframe=True,
                tracks=np.empty((0, 4), dtype=np.float32),
                num_tracks=0,
                median_pixel_motion=0.0,
                mean_pixel_motion=0.0,
                frame_id=self.frame_id,
                keyframe_id=self.keyframe_id,
            )

        prev_pts = self.last_points.reshape(-1, 1, 2).astype(np.float32)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_gray,
            gray,
            prev_pts,
            None,
            **self.lk_params,
        )

        if next_pts is None or status is None:
            self._set_new_keyframe(gray)

            return OpticalFlowResult(
                is_keyframe=True,
                tracks=np.empty((0, 4), dtype=np.float32),
                num_tracks=0,
                median_pixel_motion=0.0,
                mean_pixel_motion=0.0,
                frame_id=self.frame_id,
                keyframe_id=self.keyframe_id,
            )

        status = status.reshape(-1).astype(bool)
        next_pts = next_pts.reshape(-1, 2)

        valid = status

        if self.use_forward_backward_check:
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self.last_gray,
                next_pts.reshape(-1, 1, 2).astype(np.float32),
                None,
                **self.lk_params,
            )

            if back_pts is not None and back_status is not None:
                back_pts = back_pts.reshape(-1, 2)
                back_status = back_status.reshape(-1).astype(bool)

                fb_error = np.linalg.norm(
                    back_pts - self.last_points,
                    axis=1,
                )

                valid = valid & back_status & (fb_error < self.forward_backward_threshold)

        key_pts_valid = self.keyframe_points[valid]
        cur_pts_valid = next_pts[valid]

        tracks = np.hstack([key_pts_valid, cur_pts_valid]).astype(np.float32)

        num_tracks = len(tracks)

        if num_tracks > 0:
            displacement = np.linalg.norm(cur_pts_valid - key_pts_valid, axis=1)
            median_motion = float(np.median(displacement))
            mean_motion = float(np.mean(displacement))
        else:
            median_motion = 0.0
            mean_motion = 0.0

        is_keyframe = median_motion >= self.keyframe_pixel_threshold or num_tracks < self.min_tracked_features

        if is_keyframe:
            # Important:
            # The output tracks are still with respect to the previous keyframe.
            # After producing the result, we promote the current frame to new keyframe.
            old_keyframe_id = self.keyframe_id
            self._set_new_keyframe(gray)

            result_keyframe_id = old_keyframe_id
        else:
            # Continue tracking the same original keyframe features.
            self.last_gray = gray
            self.keyframe_points = key_pts_valid
            self.last_points = cur_pts_valid

            result_keyframe_id = self.keyframe_id

        return OpticalFlowResult(
            is_keyframe=is_keyframe,
            tracks=tracks,
            num_tracks=num_tracks,
            median_pixel_motion=median_motion,
            mean_pixel_motion=mean_motion,
            frame_id=self.frame_id,
            keyframe_id=result_keyframe_id,
        )

    def _initialize(self, gray: np.ndarray):
        self.image_shape = gray.shape
        self.initialized = True
        self._set_new_keyframe(gray)

    def _set_new_keyframe(self, gray: np.ndarray):
        self.keyframe_gray = gray
        self.last_gray = gray

        self.keyframe_id = self.frame_id

        pts = self._detect_features(gray)

        self.keyframe_points = pts
        self.last_points = pts.copy()

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        if pts is None:
            return np.empty((0, 2), dtype=np.float32)

        return pts.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            # Assumes OpenCV-style BGR input.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = gray * 255.0

            gray = np.clip(gray, 0, 255).astype(np.uint8)

        return gray


def visualize_optical_flow_result(
    image: np.ndarray,
    result,
    max_tracks: int = 300,
    draw_keyframe_points: bool = True,
    draw_current_points: bool = True,
    draw_lines: bool = True,
    draw_text: bool = True,
) -> np.ndarray:
    """
    Visualize optical-flow tracks from an OpticalFlowResult.

    Args:
        image:
            Current image, BGR / RGB / grayscale.

        result:
            OpticalFlowResult returned by processor.process(image).

        max_tracks:
            Maximum number of tracks to draw.

        draw_keyframe_points:
            Draw feature positions in the last keyframe.

        draw_current_points:
            Draw tracked feature positions in the current image.

        draw_lines:
            Draw motion lines from keyframe position to current position.

        draw_text:
            Draw tracking statistics.

    Returns:
        vis:
            BGR visualization image.

    Track format:
        result.tracks is N x 4:
            [x_keyframe, y_keyframe, x_current, y_current]
    """

    vis = _to_bgr_for_vis(image)

    tracks = result.tracks

    if tracks is None or len(tracks) == 0:
        if draw_text:
            cv2.putText(
                vis,
                "No tracks",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return vis

    tracks = tracks[:max_tracks]

    for t in tracks:
        x0, y0, x1, y1 = t

        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))

        if draw_lines:
            cv2.line(
                vis,
                p0,
                p1,
                color=(0, 255, 255),  # yellow
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        if draw_keyframe_points:
            cv2.circle(
                vis,
                p0,
                radius=2,
                color=(255, 0, 0),  # blue
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        if draw_current_points:
            cv2.circle(
                vis,
                p1,
                radius=3,
                color=(0, 255, 0),  # green
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    if draw_text:
        text_1 = f"tracks: {len(tracks)}"
        text_2 = f"median motion: {result.median_pixel_motion:.2f}px"
        text_3 = f"keyframe: {result.is_keyframe}"
        text_4 = f"frame: {result.frame_id}, ref kf: {result.keyframe_id}"

        cv2.putText(
            vis,
            text_1,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            text_2,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            text_3,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if result.is_keyframe else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            text_4,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


def _to_bgr_for_vis(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale or 3-channel image to BGR uint8 visualization image.

    Note:
        If your input is RGB, colors will be interpreted as BGR by OpenCV display.
        For tracking visualization this usually does not matter.
    """

    if image.ndim == 2:
        gray = image

        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = gray * 255.0
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3:
        img = image

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img.copy()

    raise ValueError(f"Unsupported image shape: {image.shape}")


# python da3_slam/optical_frontend.py
if __name__ == "__main__":
    processor = OpticalFlowKeyframeProcessor(
        min_feature_distance=10,
        keyframe_pixel_threshold=25.0,
        min_tracked_features=150,
    )

    import glob

    image_files = glob.glob("data/cam0-20260427T031458Z-3-001/cam0/data/*.png")
    image_files.sort()

    for image_file in image_files:
        image = cv2.imread(image_file)
        result = processor.process(image)

        if result.is_keyframe:
            print(
                f"New keyframe at frame {result.frame_id}, "
                f"tracked points = {result.num_tracks}, "
                f"median motion = {result.median_pixel_motion:.2f}px"
            )
        # tracks[:, 0:2] are feature positions in the last keyframe
        # tracks[:, 2:4] are corresponding feature positions in current frame

        vis = visualize_optical_flow_result(image, result)

        cv2.imshow("optical flow tracks", vis)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
