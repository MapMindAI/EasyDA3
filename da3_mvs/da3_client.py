import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import time
import os
import argparse

try:
    from .logging_utils import configure_logging, get_logger
    from .colmap_loader import (
        load_sparse_model,
        build_covisibility_graph,
        chunk_images_from_graph,
        get_chunk_payload,
        extrinsics_to_first_camera_frame,
    )
except ImportError:
    from logging_utils import configure_logging, get_logger
    from colmap_loader import (
        load_sparse_model,
        build_covisibility_graph,
        chunk_images_from_graph,
        get_chunk_payload,
        extrinsics_to_first_camera_frame,
    )


logger = get_logger(__name__)


def list_all_triton_models(client):
    model_index = client.get_model_repository_index()
    for model in model_index.models:
        logger.info("Model: %s, Version: %s, State: %s", model.name, model.version, model.state)


def find_triton_model(client, model_key):
    model_index = client.get_model_repository_index()
    for model in model_index.models:
        if model_key in model.name:
            return model.name
    return None


class DepthAnything3:
    def __init__(
        self,
        triton_url,
        model_key="depthanything3",
        model_version="1",
        input_height=280,
        input_width=504,
        use_imagenet_norm=False,
        near_percentile=2,
        far_percentile=98,
        gamma=0.7,
        invert_vis=True,
        expected_num_images=None,  # None means do not enforce
        input_name="image",
        intrinsics_input_name="intrinsics_in",
        extrinsics_input_name="extrinsics_in",
    ):
        self.grpc_client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.model_version = model_version
        self.model_name = find_triton_model(self.grpc_client, model_key)

        if self.model_name is None:
            raise ValueError(f"Cannot find Triton model containing key: {model_key}")

        self.input_height = input_height
        self.input_width = input_width
        self.use_imagenet_norm = use_imagenet_norm
        self.near_percentile = near_percentile
        self.far_percentile = far_percentile
        self.gamma = gamma
        self.invert_vis = invert_vis
        self.expected_num_images = expected_num_images
        self.input_name = input_name
        self.intrinsics_input_name = intrinsics_input_name
        self.extrinsics_input_name = extrinsics_input_name

        logger.info("Start %s from %s", self.model_name, triton_url)

        self.desired_outputs = [
            grpcclient.InferRequestedOutput("depth"),
            grpcclient.InferRequestedOutput("depth_conf"),
        ]

    def _preprocess_single_image(self, image_numpy):
        if image_numpy is None:
            raise ValueError("Input image is None")

        if image_numpy.ndim == 2:
            image_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_GRAY2RGB)
        elif image_numpy.ndim == 3 and image_numpy.shape[2] == 3:
            # OpenCV input is usually BGR
            image_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {image_numpy.shape}")

        orig_h, orig_w = image_rgb.shape[:2]

        image_resized = cv2.resize(
            image_rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )

        x = image_resized.astype(np.float32) / 255.0

        if self.use_imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std

        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        return image_rgb, (orig_h, orig_w), x

    def _depth_to_vis(self, depth):
        depth = depth.astype(np.float32)
        valid = np.isfinite(depth)
        if not np.any(valid):
            return np.zeros_like(depth, dtype=np.float32)

        v = depth[valid]
        dmin = np.percentile(v, self.near_percentile)
        dmax = np.percentile(v, self.far_percentile)

        depth = np.clip(depth, dmin, dmax)
        depth = (depth - dmin) / (dmax - dmin + 1e-8)

        if self.invert_vis:
            depth = 1.0 - depth

        depth = np.power(depth, self.gamma)
        return depth

    def _build_input_tensor(self, images):
        if not isinstance(images, (list, tuple)):
            raise TypeError("images must be a list or tuple of numpy images")
        if len(images) == 0:
            raise ValueError("images must contain at least one image")
        if self.expected_num_images is not None and len(images) != self.expected_num_images:
            raise ValueError(f"Expected {self.expected_num_images} images, got {len(images)}")

        image_rgbs = []
        orig_sizes = []
        tensors = []

        for img in images:
            img_rgb, orig_size, x = self._preprocess_single_image(img)
            image_rgbs.append(img_rgb)
            orig_sizes.append(orig_size)
            tensors.append(x)

        # [N, 3, H, W]
        x = np.stack(tensors, axis=0)

        # [1, N, 3, H, W]
        x = np.expand_dims(x, axis=0).astype(np.float32)

        meta = {
            "image_rgbs": image_rgbs,
            "orig_sizes": orig_sizes,
            "num_images": len(images),
        }
        return x, meta

    def get_response(self, response, key):
        result = response.as_numpy(key)
        if result is None:
            raise RuntimeError(f"Triton returned no '{key}' output")
        return result

    def _build_pose_tensors(self, intrinsics_list, extrinsics_list, num_images):
        if intrinsics_list is None or extrinsics_list is None:
            return None, None
        if len(intrinsics_list) != num_images or len(extrinsics_list) != num_images:
            raise ValueError("intrinsics_list and extrinsics_list must match number of images")

        intrinsics = []
        extrinsics = []
        for i in range(num_images):
            K = np.asarray(intrinsics_list[i], dtype=np.float32)
            E = np.asarray(extrinsics_list[i], dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(f"intrinsics_list[{i}] shape must be (3,3), got {K.shape}")
            if E.shape == (3, 4):
                E44 = np.eye(4, dtype=np.float32)
                E44[:3, :4] = E
            elif E.shape == (4, 4):
                E44 = E
            else:
                raise ValueError(f"extrinsics_list[{i}] shape must be (3,4) or (4,4), got {E.shape}")
            intrinsics.append(K)
            extrinsics.append(E44)

        K_tensor = np.expand_dims(np.stack(intrinsics, axis=0), axis=0)  # [1, N, 3, 3]
        E_tensor = np.expand_dims(np.stack(extrinsics, axis=0), axis=0)  # [1, N, 4, 4]
        return K_tensor.astype(np.float32), E_tensor.astype(np.float32)

    def _scale_intrinsics_for_resized_inputs(self, intrinsics_list, orig_sizes):
        """
        Scale intrinsics from original image resolution to resized network input resolution.
        """
        if intrinsics_list is None:
            return None
        if len(intrinsics_list) != len(orig_sizes):
            raise ValueError("intrinsics_list must match number of input images")

        scaled = []
        for i, (K_in, (orig_h, orig_w)) in enumerate(zip(intrinsics_list, orig_sizes)):
            if orig_h <= 0 or orig_w <= 0:
                raise ValueError(f"Invalid original size at index {i}: {(orig_h, orig_w)}")
            K = np.asarray(K_in, dtype=np.float32).copy()
            if K.shape != (3, 3):
                raise ValueError(f"intrinsics_list[{i}] shape must be (3,3), got {K.shape}")
            sx = float(self.input_width) / float(orig_w)
            sy = float(self.input_height) / float(orig_h)
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            scaled.append(K)
        return scaled

    def run(self, images, intrinsics_list=None, extrinsics_list=None):
        x, meta = self._build_input_tensor(images)
        intrinsics_for_input = self._scale_intrinsics_for_resized_inputs(
            intrinsics_list=intrinsics_list,
            orig_sizes=meta["orig_sizes"],
        )

        inputs = []
        input_tensor = grpcclient.InferInput(self.input_name, x.shape, "FP32")
        input_tensor.set_data_from_numpy(x)
        inputs.append(input_tensor)
        K_tensor, E_tensor = self._build_pose_tensors(
            intrinsics_list=intrinsics_for_input,
            extrinsics_list=extrinsics_list,
            num_images=meta["num_images"],
        )
        if K_tensor is not None and E_tensor is not None:
            k_input = grpcclient.InferInput(self.intrinsics_input_name, K_tensor.shape, "FP32")
            e_input = grpcclient.InferInput(self.extrinsics_input_name, E_tensor.shape, "FP32")
            k_input.set_data_from_numpy(K_tensor)
            e_input.set_data_from_numpy(E_tensor)
            inputs.extend([k_input, e_input])

        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=self.desired_outputs,
        )

        depth = self.get_response(response, "depth")
        depth_conf = self.get_response(response, "depth_conf")

        # expected [1, N, H, W]
        if depth.ndim != 4 or depth.shape[0] != 1:
            raise ValueError(f"Unexpected depth output shape: {depth.shape}")

        num_images = meta["num_images"]
        if depth.shape[1] != num_images:
            raise ValueError(f"Output image count mismatch: input has {num_images}, output has {depth.shape[1]}")
        if depth_conf.shape != depth.shape:
            raise ValueError(f"Unexpected depth_conf output shape: {depth_conf.shape}, expected {depth.shape}")

        depth_list = []
        depth_conf_list = []

        for i in range(num_images):
            depth_i = depth[0, i]
            depth_conf_i = depth_conf[0, i]
            orig_h, orig_w = meta["orig_sizes"][i]
            depth_i_resized = cv2.resize(depth_i, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            depth_conf_i_resized = cv2.resize(depth_conf_i, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            depth_list.append(depth_i_resized)
            depth_conf_list.append(depth_conf_i_resized.astype(np.float32))

        result = {
            "depth_list": depth_list,  # list of HxW
            "depth_conf_list": depth_conf_list,  # list of HxW
        }
        if intrinsics_list is not None:
            result["intrinsics_list"] = intrinsics_list
        if extrinsics_list is not None:
            result["extrinsics_list"] = extrinsics_list
        return result

    def run_paths(self, image_paths):
        if not isinstance(image_paths, (list, tuple)):
            raise TypeError("image_paths must be a list or tuple")

        images = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {p}")
            images.append(img)

        return self.run(images), images

    def save_visualizations(self, result, prefix="depth"):
        for i, depth_i in enumerate(result["depth_list"]):
            depth_i_vis = self._depth_to_vis(depth_i)
            conf_i = None
            if "depth_conf_list" in result and i < len(result["depth_conf_list"]):
                conf_i = np.asarray(result["depth_conf_list"][i], dtype=np.float32)

            depth_u8 = (np.clip(depth_i_vis, 0, 1) * 255).astype(np.uint8)
            depth_save_path = f"{prefix}_{i}_depth_vis.png"
            cv2.imwrite(depth_save_path, depth_u8)
            logger.info("Saved %s", depth_save_path)

            if conf_i is not None:
                conf_u8 = (np.clip(conf_i, 0.0, 10.0) * 25).astype(np.uint8)
                conf_save_path = f"{prefix}_{i}_conf_vis.png"
                cv2.imwrite(conf_save_path, conf_u8)
                logger.info("Saved %s", conf_save_path)

                vis = np.concatenate([depth_u8, conf_u8], axis=1)
                pair_save_path = f"{prefix}_{i}_pair_vis.png"
                cv2.imwrite(pair_save_path, vis)
                logger.info("Saved %s", pair_save_path)
            else:
                pair_save_path = f"{prefix}_{i}_pair_vis.png"
                cv2.imwrite(pair_save_path, depth_u8)
                logger.info("Saved %s", pair_save_path)


"""
python da3_mvs/da3_client.py \
--triton-url 0.0.0.0:8001 \
--colmap-model-dir ../EasyGaussianSplatting/data/insta360_test/sparse/0 \
--image-dir ../EasyGaussianSplatting/data/insta360_test/images \
--output-dir ../EasyGaussianSplatting/data/insta360_test/da3 \
--chunk-size 10 \
--min-shared-points 20
"""

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Run DepthAnything3 Triton client over a full COLMAP sparse model and save depth/conf."
    )
    parser.add_argument("--triton-url", type=str, default="0.0.0.0:8001")
    parser.add_argument("--colmap-model-dir", type=str, required=True, help="Path to COLMAP sparse model dir")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing COLMAP images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--min-shared-points", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    depth_dir = os.path.join(args.output_dir, "depth")
    conf_dir = os.path.join(args.output_dir, "depth_conf")
    pair_viz_dir = os.path.join(args.output_dir, "depth_pair_viz")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(pair_viz_dir, exist_ok=True)

    da3 = DepthAnything3(
        triton_url=args.triton_url,
        expected_num_images=None,
    )

    images, cameras, points3D = load_sparse_model(args.colmap_model_dir)
    graph = build_covisibility_graph(images, points3D, min_shared_points=args.min_shared_points)
    chunks = chunk_images_from_graph(images, graph, chunk_size=args.chunk_size)
    logger.info("Loaded model: %d images, %d chunks", len(images), len(chunks))

    total_start_ms = time.time() * 1000.0
    written = 0
    for chunk_idx, chunk in enumerate(chunks):
        payload = get_chunk_payload(images, cameras, chunk)
        image_paths = [os.path.join(args.image_dir, n) for n in payload["image_names"]]
        img_list = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {p}")
            img_list.append(img)

        chunk_start_ms = time.time() * 1000.0
        result = da3.run(
            img_list,
            intrinsics_list=payload["intrinsics_list"],
            extrinsics_list=extrinsics_to_first_camera_frame(payload["extrinsics_list"]),
        )
        chunk_end_ms = time.time() * 1000.0
        logger.info(
            "Chunk %d/%d processed in %.3fms (size=%d)",
            chunk_idx + 1,
            len(chunks),
            chunk_end_ms - chunk_start_ms,
            len(chunk),
        )

        for image_name, depth_i, conf_i in zip(
            payload["image_names"],
            result["depth_list"],
            result["depth_conf_list"],
        ):
            stem = os.path.splitext(image_name)[0]
            safe_stem = stem.replace("\\", "_").replace("/", "_")
            np.save(os.path.join(depth_dir, f"{safe_stem}.npy"), depth_i.astype(np.float32))
            np.save(os.path.join(conf_dir, f"{safe_stem}.npy"), conf_i.astype(np.float32))

            depth_vis = da3._depth_to_vis(depth_i)
            depth_u8 = (np.clip(depth_vis, 0.0, 1.0) * 255).astype(np.uint8)
            conf_u8 = (np.clip(conf_i, 0.0, 10.0) * 25).astype(np.uint8)
            pair_u8 = np.concatenate([depth_u8, conf_u8], axis=1)
            cv2.imwrite(os.path.join(pair_viz_dir, f"{safe_stem}.png"), pair_u8)
            written += 1

    total_end_ms = time.time() * 1000.0
    logger.info("Done. Wrote %d depth/depth_conf pairs to %s in %.3fms", written, args.output_dir, total_end_ms - total_start_ms)
