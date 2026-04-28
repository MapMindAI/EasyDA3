import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import time


def list_all_triton_models(client):
    model_index = client.get_model_repository_index()
    for model in model_index.models:
        print(f"Model: {model.name}, Version: {model.version}, State: {model.state}")


def find_triton_model(client, model_key):
    model_index = client.get_model_repository_index()
    for model in model_index.models:
        if model_key in model.name:
            return model.name
    return None


def merge_rgbd_to_pointcloud_numpy(
    depth_list,
    intrinsics_list,
    extrinsics_list,
    rgb_list=None,
    depth_scale=1.0,
    min_depth=1e-6,
    max_depth=None,
    sample_stride=1,
):
    """
    Merge multiple RGBD frames into one world-space point cloud using only NumPy.

    Args:
        depth_list: list of depth maps, each shape (H, W)
        intrinsics_list: list of camera intrinsics, each shape (3, 3)
        extrinsics_list: list of extrinsics, each shape (3, 4) or (4, 4),
            assumed to be world-to-camera (w2c)
        rgb_list: optional list of RGB images, each shape (H, W, 3), dtype uint8 or float
        depth_scale: multiply raw depth by this value
        min_depth: discard points with depth <= min_depth
        max_depth: optional upper bound for depth filtering
        sample_stride: keep one pixel every `sample_stride`

    Returns:
        points: (N, 3) float32 world-space points
        colors: (N, 3) uint8 colors if rgb_list is provided, else None
    """
    if not (len(depth_list) == len(intrinsics_list) == len(extrinsics_list)):
        raise ValueError("depth_list, intrinsics_list, extrinsics_list must have same length")

    if rgb_list is not None and len(rgb_list) != len(depth_list):
        raise ValueError("rgb_list must have same length as depth_list")

    all_points = []
    all_colors = []

    for i in range(len(depth_list)):
        depth = np.asarray(depth_list[i], dtype=np.float32) * depth_scale
        K = np.asarray(intrinsics_list[i], dtype=np.float32)
        ext = np.asarray(extrinsics_list[i], dtype=np.float32)

        if depth.ndim != 2:
            raise ValueError(f"depth_list[{i}] must have shape (H, W), got {depth.shape}")
        if K.shape != (3, 3):
            raise ValueError(f"intrinsics_list[{i}] must have shape (3, 3), got {K.shape}")
        if ext.shape == (3, 4):
            ext44 = np.eye(4, dtype=np.float32)
            ext44[:3, :4] = ext
        elif ext.shape == (4, 4):
            ext44 = ext
        else:
            raise ValueError(f"extrinsics_list[{i}] must have shape (3, 4) or (4, 4), got {ext.shape}")

        H, W = depth.shape

        if rgb_list is not None:
            rgb = np.asarray(rgb_list[i])
            if rgb.shape[:2] != (H, W) or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"rgb_list[{i}] must have shape (H, W, 3) matching depth, got {rgb.shape}")
            if rgb.dtype != np.uint8:
                if np.issubdtype(rgb.dtype, np.floating):
                    rgb = np.clip(rgb, 0.0, 1.0)
                    rgb = (rgb * 255.0).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # pixel grid
        ys, xs = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij",
        )

        if sample_stride > 1:
            stride_mask = np.zeros((H, W), dtype=bool)
            stride_mask[::sample_stride, ::sample_stride] = True
        else:
            stride_mask = np.ones((H, W), dtype=bool)

        valid = np.isfinite(depth) & (depth > min_depth) & stride_mask
        if max_depth is not None:
            valid &= depth < max_depth

        if not np.any(valid):
            continue

        z = depth[valid]
        u = xs[valid]
        v = ys[valid]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # back-project to camera coordinates
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)

        # camera-to-world = inverse(world-to-camera)
        c2w = np.linalg.inv(ext44)

        pts_cam_h = np.concatenate(
            [pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)],
            axis=1,
        )  # (N, 4)

        pts_world_h = pts_cam_h @ c2w.T
        pts_world = pts_world_h[:, :3].astype(np.float32)

        all_points.append(pts_world)

        if rgb_list is not None:
            cols = rgb[valid].reshape(-1, 3)
            all_colors.append(cols)

    if len(all_points) == 0:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_colors = np.zeros((0, 3), dtype=np.uint8) if rgb_list is not None else None
        return empty_points, empty_colors

    points = np.concatenate(all_points, axis=0)

    if rgb_list is not None:
        colors = np.concatenate(all_colors, axis=0)
    else:
        colors = None

    return points, colors


def save_pointcloud_ply(path, points, colors=None):
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if len(colors) != len(points):
            raise ValueError("colors and points must have same length")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors is None:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


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

        print(f"Start {self.model_name} from {triton_url}")

        self.desired_outputs = [
            grpcclient.InferRequestedOutput("depth"),
            grpcclient.InferRequestedOutput("intrinsics"),
            grpcclient.InferRequestedOutput("extrinsics"),
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

    def run(self, images):
        x, meta = self._build_input_tensor(images)

        inputs = []
        input_tensor = grpcclient.InferInput(self.input_name, x.shape, "FP32")
        input_tensor.set_data_from_numpy(x)
        inputs.append(input_tensor)

        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=self.desired_outputs,
        )

        depth = self.get_response(response, "depth")
        extrinsics = self.get_response(response, "extrinsics")
        intrinsics = self.get_response(response, "intrinsics")

        # expected [1, N, H, W]
        if depth.ndim != 4 or depth.shape[0] != 1:
            raise ValueError(f"Unexpected depth output shape: {depth.shape}")

        num_images = meta["num_images"]
        if depth.shape[1] != num_images:
            raise ValueError(f"Output image count mismatch: input has {num_images}, output has {depth.shape[1]}")

        depth_list = []
        intrinsics_list = []
        extrinsics_list = []

        for i in range(num_images):
            depth_i = depth[0, i]
            orig_h, orig_w = meta["orig_sizes"][i]
            factor_x = orig_w / depth_i.shape[1]
            factor_y = orig_h / depth_i.shape[0]
            depth_i_resized = cv2.resize(depth_i, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            depth_list.append(depth_i_resized)
            intri = intrinsics[0, i].copy()
            intri[0, 0] *= factor_x
            intri[0, 2] *= factor_x
            # TODO: the image might be distorted, we use x focus currently
            # we better keep the image ratio while input
            intri[1, 1] *= factor_x
            # intri[1, 1] *= factor_y
            intri[1, 2] *= factor_y
            intrinsics_list.append(intri)
            extrinsics_list.append(extrinsics[0, i])

        return {
            "depth_list": depth_list,  # list of HxW
            "intrinsics_list": intrinsics_list,  # list of 3x3
            "extrinsics_list": extrinsics_list,  # list of 3x4
        }

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
            save_path = f"{prefix}_{i}.png"
            depth_u8 = (np.clip(depth_i_vis, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(save_path, depth_u8)
            print(f"Saved {save_path}")


if __name__ == "__main__":
    da3 = DepthAnything3(
        triton_url="0.0.0.0:8001",
        expected_num_images=None,
    )

    import glob

    image_paths = glob.glob("data/da3/*.jpg")
    image_paths += glob.glob("data/da3/*.png")
    image_paths.sort()

    time_ms_begin = time.time() * 1000
    result, images = da3.run_paths(image_paths)
    time_ms_end = time.time() * 1000
    print(f"DA3 used {time_ms_end - time_ms_begin}ms")

    da3.save_visualizations(result, prefix="data/depth")

    print("Generating point cloud to file")
    points, colors = merge_rgbd_to_pointcloud_numpy(
        depth_list=result["depth_list"],
        intrinsics_list=result["intrinsics_list"],
        extrinsics_list=result["extrinsics_list"],
        rgb_list=images,
        min_depth=0.01,
        max_depth=100.0,
        sample_stride=4,
    )
    save_pointcloud_ply("data/cloud.ply", points, colors)
