import blenderproc as bproc

"""
Render script to render scenes using BlenderProc.
Usage: 
    blenderproc run ./scripts/render_blender.py \
    --sdf_folder /path/to/sdfs \
    --output_folder ./output_dir \
    --all_scenes
"""

import argparse
import logging
import os
import numpy as np
from typing import List, Optional
from pysdf import SDF
from scipy.spatial.transform import Rotation
from urllib.parse import urlparse

# NOTE: `bpy` is only available when this script is executed via `blenderproc run ...`
try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None

# Configure logging
logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger(__name__)


# --- Helper: Math & Transformations ---

def get_matrix_from_xyz_euler(xyz_euler: Optional[List[float]]) -> np.ndarray:
    """
    Converts a list of [x, y, z, roll, pitch, yaw] to a 4x4 transformation matrix.
    """
    if xyz_euler is None or len(xyz_euler) != 6:
        return np.eye(4)
    
    x, y, z, roll, pitch, yaw = xyz_euler
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    tmat = np.eye(4)
    tmat[0:3, 0:3] = rotation
    tmat[0:3, 3] = [x, y, z]
    return tmat


def get_xyz_euler_from_matrix(tmat: np.ndarray) -> List[float]:
    """
    Converts a 4x4 transformation matrix to a list of [x, y, z, roll, pitch, yaw].
    """
    x, y, z = tmat[0:3, 3]
    rotation = Rotation.from_matrix(tmat[0:3, 0:3])
    roll, pitch, yaw = rotation.as_euler('xyz')
    return [x, y, z, roll, pitch, yaw]


def resolve_asset_uri(uri: str, mesh_root: str) -> str:
    """
    Resolve common SDFormat mesh URI formats to a local filesystem path.

    Supported:
    - package://foo/bar.obj  -> <mesh_root>/foo/bar.obj
    - file:///abs/path.obj   -> /abs/path.obj
    - /abs/path.obj          -> /abs/path.obj
    - relative/path.obj      -> <mesh_root>/relative/path.obj
    """
    if uri is None:
        return ""

    uri_str = str(uri).strip()
    if not uri_str:
        return ""

    mesh_root_abs = os.path.abspath(mesh_root)

    if uri_str.startswith("package://"):
        rel = uri_str[len("package://") :].lstrip("/")
        return os.path.normpath(os.path.join(mesh_root_abs, rel))

    if uri_str.startswith("file://"):
        parsed = urlparse(uri_str)
        if parsed.path:
            return os.path.normpath(parsed.path)
        return os.path.normpath(uri_str[len("file://") :])

    if os.path.isabs(uri_str):
        return os.path.normpath(uri_str)

    return os.path.normpath(os.path.join(mesh_root_abs, uri_str))


# --- Helper: Scene Setup ---

def setup_lighting():
    """Sets up a standard point light for the scene."""
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([3, -2, 3])
    light.set_energy(1000)


def setup_cameras(n_poses: int, width: int = 1200, height: int = 1200) -> List[np.ndarray]:
    """
    Generates random camera poses looking downwards at the scene center.
    Returns list of intrinsic matrices (K).
    """
    camera_intrinsics = []
    
    fx = 1000
    fy = 1000
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Set intrinsics once (assuming constant for all views)
    focal_length_mm = fx * (36 / width) # Assuming 36mm sensor width
    bproc.camera.set_intrinsics_from_K_matrix(K, width, height)
    bproc.camera.set_intrinsics_from_blender_params(lens=focal_length_mm, lens_unit="MILLIMETERS")

    for _ in range(n_poses):
        # Randomize Pose
        # x between 0.9 and 1.2, z between -0.1 and 0.2
        pos_x = np.random.uniform(1.2, 1.3) 
        pos_z = np.random.uniform(-0.1, 0.1)
        
        # Looking generally downwards
        yaw = np.random.uniform(np.pi * 0.5 - 0.1, np.pi * 0.5 + 0.1)
        roll = np.random.uniform(np.pi * 0.5 - 0.1, np.pi * 0.5 + 0.1)
        
        # Construct camera matrix (Position + Rotation)
        # Note: Your original code used [x, 0, z] for position. 
        # Ensure this matches your scene's coordinate system (e.g., Y-up vs Z-up).
        camera_pose_euler = [pos_x, 0, pos_z, roll, 0, yaw]
        
        # Convert Euler to Matrix
        rotation_matrix = Rotation.from_euler('xyz', camera_pose_euler[3:]).as_matrix()
        cam2world = np.eye(4)
        cam2world[0:3, 0:3] = rotation_matrix
        cam2world[0:3, 3] = camera_pose_euler[:3]

        bproc.camera.add_camera_pose(cam2world)
        camera_intrinsics.append(K)
        
    return camera_intrinsics


def load_sdf_into_blender(sdf_path: str, mesh_root: str = "./data") -> None:
    """
    Parses an SDF file and loads all visuals into BlenderProc, applying 
    hierarchical transformations (Model -> Link -> Visual).
    """
    parsed = SDF.from_file(sdf_path)
    world = parsed.worlds[0] if parsed.worlds else None
    
    if world is None:
        console_logger.warning(f"No world found in SDF: {sdf_path}")
        return

    console_logger.info(f"Loading SDF: {sdf_path}")

    for model in world.models:
        # 1. Model Transform
        T_world_model = get_matrix_from_xyz_euler(model.pose.value)

        for link in model.links:
            # 2. Link Transform (relative to Model)
            T_model_link = get_matrix_from_xyz_euler(link.pose.value)
            
            # Combine: T_world_link = T_world_model * T_model_link
            T_world_link = np.dot(T_world_model, T_model_link)

            for visual in link.visuals:
                # 3. Visual Transform (relative to Link)
                T_link_visual = get_matrix_from_xyz_euler(visual.pose.value)
                
                # Combine: T_world_visual = T_world_link * T_link_visual
                T_final = np.dot(T_world_link, T_link_visual)

                # Get final Euler/Location for Blender
                xyz_euler = get_xyz_euler_from_matrix(T_final)
                scale = visual.geometry.mesh.scale if visual.geometry.mesh.scale else [1.0, 1.0, 1.0]

                # Load Mesh
                mesh_uri = visual.geometry.mesh.uri
                abs_mesh_path = resolve_asset_uri(mesh_uri, mesh_root)
                
                if not os.path.exists(abs_mesh_path):
                    console_logger.error(
                        f"Mesh not found. uri='{mesh_uri}' resolved='{abs_mesh_path}' mesh_root='{os.path.abspath(mesh_root)}'"
                    )
                    continue

                loaded_objs = bproc.loader.load_obj(abs_mesh_path)
                
                for obj in loaded_objs:
                    obj.set_location([xyz_euler[0], xyz_euler[1], xyz_euler[2]])
                    obj.set_rotation_euler([xyz_euler[3], xyz_euler[4], xyz_euler[5]])
                    obj.set_scale(scale)
                    
                    # Set Custom Properties for Segmentation
                    obj.set_cp("mesh_file", mesh_uri)


def _configure_cycles(
    device: str,
    compute_device_type: str,
    gpu_indices: Optional[List[int]],
    width: int,
    height: int,
    samples: int,
) -> None:
    """
    Configure Cycles device selection and basic render settings.

    Important: Cycles multi-GPU duplicates scene data on each enabled GPU. On shared nodes, if Blender
    enables all GPUs and any one of them is low on free VRAM, you can get "System is out of GPU memory"
    even if other GPUs have plenty of free memory.
    """
    if bpy is None:
        console_logger.warning("`bpy` not available; cannot configure Cycles devices/settings.")
        return

    scene = bpy.context.scene

    # Resolution affects per-frame buffers and can materially change memory usage.
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100

    # Samples affect runtime more than memory, but make it configurable.
    if hasattr(scene, "cycles"):
        scene.cycles.samples = int(samples)
        # Persistent data can increase memory; safer default for batch runs.
        if hasattr(scene.cycles, "use_persistent_data"):
            scene.cycles.use_persistent_data = False

    device = (device or "gpu").strip().lower()
    if device == "cpu":
        if hasattr(scene, "cycles"):
            scene.cycles.device = "CPU"
        console_logger.info("Cycles device set to CPU.")
        return

    # GPU path: configure Cycles addon prefs to use only selected GPUs.
    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons["cycles"].preferences
    except Exception as e:  # pragma: no cover
        console_logger.warning(f"Could not access Cycles addon preferences; leaving default device selection. ({e})")
        if hasattr(scene, "cycles"):
            scene.cycles.device = "GPU"
        return

    compute_device_type = (compute_device_type or "OPTIX").strip().upper()
    try:
        cycles_prefs.compute_device_type = compute_device_type
    except Exception as e:
        console_logger.warning(
            f"Could not set Cycles compute_device_type={compute_device_type}; leaving default. ({e})"
        )

    # Populate device list
    try:
        cycles_prefs.get_devices()
    except Exception:
        pass

    devices = list(getattr(cycles_prefs, "devices", []))
    for d in devices:
        try:
            d.use = False
        except Exception:
            pass

    enabled = 0
    if gpu_indices:
        for idx in gpu_indices:
            if 0 <= idx < len(devices):
                try:
                    devices[idx].use = True
                    enabled += 1
                except Exception:
                    pass
            else:
                console_logger.warning(
                    f"Requested GPU index {idx} is out of range (0..{max(len(devices)-1, 0)})."
                )
    else:
        # If user didn't specify indices, enable all GPUs (keeps existing behavior).
        for d in devices:
            if getattr(d, "type", "").upper() == "CPU":
                continue
            try:
                d.use = True
                enabled += 1
            except Exception:
                pass

    if hasattr(scene, "cycles"):
        scene.cycles.device = "GPU"

    if gpu_indices:
        console_logger.info(
            f"Cycles device set to GPU ({compute_device_type}); enabled {enabled} device(s) from indices={gpu_indices}."
        )
    else:
        console_logger.info(
            f"Cycles device set to GPU ({compute_device_type}); enabled {enabled} device(s) (all GPUs)."
        )


# --- Main Pipeline ---

def render_scene(
    sdf_path: str,
    output_dir: str,
    n_camera_poses: int = 2,
    mesh_root: str = "./data",
    width: int = 1200,
    height: int = 1200,
):
    """
    Orchestrates the rendering for a single SDF scene.
    """
    # Clear previous scene data
    bproc.clean_up()
    
    # 1. Setup Camera & Light
    camera_intrinsics = setup_cameras(n_poses=n_camera_poses, width=width, height=height)
    setup_lighting()
    
    # 2. Load Scene Geometry
    load_sdf_into_blender(sdf_path, mesh_root=mesh_root)
    
    # 3. Configure Output Data
    bproc.renderer.enable_segmentation_output(
        map_by=["instance", "name", "mesh_file"], 
        default_values={'mesh_file': None}
    )

    # 4. Render
    console_logger.info("Rendering...")
    data = bproc.renderer.render()
    
    # 5. Post-Processing (Point Clouds)
    all_point_clouds = []
    for idx, depth_map in enumerate(data['depth']):
        points = bproc.camera.pointcloud_from_depth(depth_map, idx)
        all_point_clouds.append(points)
    
    data['point_clouds'] = all_point_clouds
    data['camera_intrinsics'] = camera_intrinsics

    # 6. Save
    os.makedirs(output_dir, exist_ok=True)
    bproc.writer.write_hdf5(output_dir, data)
    console_logger.info(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render a Blender scene from SDF.")
    parser.add_argument("--sdf_folder", type=str, required=True, help="Path to folder containing SDF files.")
    parser.add_argument("--output_folder", type=str, default="./render_output", help="Root output folder.")
    parser.add_argument("--all_scenes", action="store_true", help="If set, render all scenes in folder.")
    parser.add_argument(
        "--mesh_root",
        type=str,
        default="./data",
        help="Root directory containing asset packages referenced by package:// URIs (e.g. <mesh_root>/gazebo, <mesh_root>/tri).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Cycles render device. Use 'cpu' to avoid GPU OOM on shared nodes (slower).",
    )
    parser.add_argument(
        "--compute_device_type",
        type=str,
        default="OPTIX",
        help="Cycles compute device type (e.g. OPTIX, CUDA). Only used when --device gpu.",
    )
    parser.add_argument(
        "--gpu_indices",
        type=str,
        default="6",
        help="Comma-separated GPU indices to enable inside Blender (e.g. '6' or '0,1'). "
             "If empty, enables all GPUs (default behavior).",
    )
    parser.add_argument("--width", type=int, default=1200, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=1200, help="Render height in pixels.")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel.")
    
    args = parser.parse_args()
    
    # Initialize BlenderProc (run once)
    bproc.init()
    gpu_indices: Optional[List[int]] = None
    if args.gpu_indices.strip():
        try:
            gpu_indices = [int(x.strip()) for x in args.gpu_indices.split(",") if x.strip() != ""]
        except ValueError:
            raise ValueError(f"Invalid --gpu_indices='{args.gpu_indices}'. Expected comma-separated integers.")

    _configure_cycles(
        device=args.device,
        compute_device_type=args.compute_device_type,
        gpu_indices=gpu_indices,
        width=args.width,
        height=args.height,
        samples=args.samples,
    )
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    
    # Gather SDF files
    sdf_files = sorted([f for f in os.listdir(args.sdf_folder) if f.endswith(".sdf")])
    if not args.all_scenes and sdf_files:
        sdf_files = sdf_files[:1]

    console_logger.info(f"Found {len(sdf_files)} SDF files to process.")

    # Process each scene
    for sdf_file in sdf_files:
        sdf_path = os.path.join(args.sdf_folder, sdf_file)
        
        # Create a unique output subfolder for this scene
        scene_name = os.path.splitext(sdf_file)[0]
        scene_output_dir = os.path.join(args.output_folder, scene_name)
        
        render_scene(
            sdf_path,
            scene_output_dir,
            n_camera_poses=1,
            mesh_root=args.mesh_root,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()