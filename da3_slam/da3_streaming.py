from __future__ import annotations

import cv2

try:
    from .backend.da3_pose_graph_optimizer import DA3ChunkPoseGraphOptimizer
    from .da3_client import DepthAnything3
    from .logging_utils import configure_logging, get_logger
    from .optical_frontend import (
        OpticalFlowKeyframeProcessor,
        create_optical_flow_processor,
        visualize_optical_flow_result,
    )
    from .streaming import (
        DA3BackendJob,
        DA3BackendResult,
        DA3ChunkRecord,
        DA3StreamingMappingPipeline,
        KeyframeRecord,
        MappingProcessResult,
    )
    from .visualizer import Open3DTrajectoryVisualizer
except ImportError:
    from backend.da3_pose_graph_optimizer import DA3ChunkPoseGraphOptimizer
    from da3_client import DepthAnything3
    from logging_utils import configure_logging, get_logger
    from optical_frontend import (
        OpticalFlowKeyframeProcessor,
        create_optical_flow_processor,
        visualize_optical_flow_result,
    )
    from streaming import (
        DA3BackendJob,
        DA3BackendResult,
        DA3ChunkRecord,
        DA3StreamingMappingPipeline,
        KeyframeRecord,
        MappingProcessResult,
    )
    from visualizer import Open3DTrajectoryVisualizer


logger = get_logger(__name__)

__all__ = [
    "DA3BackendJob",
    "DA3BackendResult",
    "DA3ChunkRecord",
    "DA3StreamingMappingPipeline",
    "KeyframeRecord",
    "MappingProcessResult",
    "OpticalFlowKeyframeProcessor",
    "create_optical_flow_processor",
]


def build_default_pipeline() -> DA3StreamingMappingPipeline:
    flow_processor = create_optical_flow_processor("loose")

    da3_client = DepthAnything3(
        triton_url="0.0.0.0:8001",
        expected_num_images=None,
    )

    pose_optimizer = DA3ChunkPoseGraphOptimizer(
        input_pose_is_w2c=True,
        relative_rotation_sigma=0.03,
        relative_translation_sigma=0.05,
        max_iterations=100,
    )

    return DA3StreamingMappingPipeline(
        flow_processor=flow_processor,
        da3_client=da3_client,
        pose_graph_optimizer=pose_optimizer,
        output_dir="output_da3_map",
        window_size=10,
        da3_stride_new_keyframes=5,
        optimizer_num_chunks=3,
    )


def resize_image(image, max_width=640):
    """
    Resize image while keeping aspect ratio.
    Only resizes if image width is greater than max_width.
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_width = max_width
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def main() -> None:
    configure_logging()

    pipeline = build_default_pipeline()

    traj_vis = Open3DTrajectoryVisualizer(
        update_hz=5.0,
        window_title="DA3-SLAM Open3D Trajectory",
        camera_size=0.25,
    )
    traj_vis.start()

    import glob

    image_files = glob.glob("data/AmsterdamMorningDrive/*.png")
    image_files.sort()

    quit_requested = False

    try:
        for image_file in image_files:
            image = cv2.imread(image_file)
            image = resize_image(image)

            result = pipeline.process_image(image)
            traj_vis.update_from_pipeline(pipeline)

            if result.is_keyframe:
                logger.info("New keyframe: %s", result.keyframe_id)

            if result.da3_scheduled:
                logger.info("DA3 chunk scheduled: %s", result.scheduled_chunk_id)

            if result.da3_ran:
                logger.info("DA3 chunk completed: %s", result.new_chunk_id)

            if result.optimization_ran:
                logger.info("Optimized images: %s", result.optimized_image_ids)

            if result.backend_error:
                logger.error("DA3 backend error: %s", result.backend_error)

            vis = visualize_optical_flow_result(image, result.flow_result)

            cv2.imshow("optical flow tracks", vis)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                quit_requested = True
                break
    finally:
        pipeline.close(wait=not quit_requested, drain=not quit_requested)
        cv2.destroyAllWindows()
        traj_vis.stop()


if __name__ == "__main__":
    main()
