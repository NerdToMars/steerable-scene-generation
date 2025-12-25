"""
Script to export scenes to SDFormat for BlenderProc.
Usage: python ./scripts/export_sdformat.py <dataset_path>
"""

import argparse
import logging
import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from pysdf import SDF, World

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.hf_dataset import (
    get_scene_vec_description_from_metadata,
    load_hf_dataset_with_metadata,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger(__name__)


def parse_models_drake_sdf(sdf_path: str) -> list:
    """
    Parses an SDF file, handling the specific Drake namespace if required.
    """
    with open(sdf_path, 'r') as f:
        sdf_string = f.read()
        
    # Inject namespace if missing to avoid parsing errors
    if 'xmlns:drake' not in sdf_string:
        sdf_string = sdf_string.replace(
            '<sdf version="', 
            '<sdf xmlns:drake="drake.mit.edu" version="'
        )
        
    parsed = SDF.from_xml(sdf_string)
    models = [parsed.model] if parsed.model else parsed.models
    return models


def resolve_model_path(raw_path: str, root_path: str) -> str:
    """Resolves package paths to absolute system paths."""
    if raw_path.startswith("package://"):
        clean_path = raw_path.replace("package://", "")
    else:
        clean_path = raw_path
        
    return os.path.join(root_path, clean_path)


def convert_rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix to XYZ Euler angles in radians."""
    rotation_obj = Rotation.from_matrix(rotation_matrix)
    return rotation_obj.as_euler('xyz')


def save_scene_sdf_for_blender_proc(
    sdf_path: str,
    scene: np.ndarray,
    scene_vec_desc: SceneVecDescription,
    root_path: str = "./data/",
) -> None:
    """
    Converts a scene vector into an SDF file compatible with BlenderProc.

    Args:
        sdf_path: Output path for the SDF file.
        scene: Scene tensor (un-normalized).
        scene_vec_desc: Description of the scene vector structure.
        root_path: Root path for model assets.
    """
    # Ensure float32 for compatibility
    if isinstance(scene, torch.Tensor) and scene.dtype == torch.bfloat16:
        scene = scene.to(torch.float32)

    world = World()
    model_name_dict = {}

    for i, obj in enumerate(scene):
        # 1. Retrieve Model Path
        idx_or_obj = i if scene_vec_desc.model_path_vec_len is None else obj
        raw_model_path = scene_vec_desc.get_model_path(idx_or_obj)

        if raw_model_path is None or not raw_model_path.endswith(".sdf"):
            continue

        model_path = resolve_model_path(raw_model_path, root_path)

        # 2. Retrieve Pose Information
        translation = scene_vec_desc.get_translation_vec(obj)
        rotation_matrix = (
            scene_vec_desc.get_rotation_matrix(torch.tensor(obj))
            .to(torch.float32)
            .numpy()
        )
        euler_angles_rad = convert_rotation_matrix_to_euler(rotation_matrix)

        # 3. Parse and update SDF models
        if model_path.endswith('.sdf'):
            for model in parse_models_drake_sdf(model_path):
                # Update Pose text (x y z roll pitch yaw)
                model.pose.text = "{} {} {} {} {} {}".format(
                    translation[0], translation[1], translation[2],
                    euler_angles_rad[0], euler_angles_rad[1], euler_angles_rad[2]
                )

                # Update Mesh URIs to absolute paths
                for link in model.links:
                    for visual in link.visuals:
                        mesh_file = visual.geometry.mesh.uri
                        if mesh_file.startswith("package://"):
                            mesh_file = mesh_file[len("package://"):]
                        
                        base_folder = raw_model_path.rsplit("/", 1)[0] + "/"
                        abs_mesh_file = base_folder + mesh_file
                        visual.geometry.mesh.uri = abs_mesh_file

                # Handle name collisions
                if model.name not in model_name_dict:
                    model_name_dict[model.name] = 0
                else:
                    model_name_dict[model.name] += 1
                
                model.name = f"{model.name}_{model_name_dict[model.name]}"
                world.add(model)

    # Save final SDF
    sdf_f = SDF(world)
    sdf_f.to_file(sdf_path, pretty_print=True)


def main():
    parser = argparse.ArgumentParser(description="Export scenes to SDF for BlenderProc.")
    parser.add_argument(
        "scenes_data_path", type=str, help="Path to the scene HF dataset folder."
    )
    parser.add_argument(
        "--output_sdf_folder", type=str, default="./outputs", help="Path to save the exported SDF files."
    )
    parser.add_argument(
        "--scene_idx", default=0, type=int, help="Index of the scene to visualize start from."
    )
    parser.add_argument(
        "--filter_string",
        type=str,
        default=None,
        help="String to filter scenes by their language_annotation field.",
    )
    parser.add_argument(
        "--not_weld_objects",
        action="store_true",
        help="Do not weld the objects to the world frame.",
    )
    parser.add_argument(
        "--visualize_proximity",
        action="store_true",
        help="Whether to visualize proximity.",
    )
    parser.add_argument(
        "--package_names",
        nargs="+",
        type=str,
        default=["tri", "gazebo", "greg"],
        help="Optional list of package names for resolving model paths.",
    )
    parser.add_argument(
        "--package_file_paths",
        nargs="+",
        type=str,
        default=[
            "data/tri/package.xml",
            "data/gazebo/package.xml",
            "data/greg/package.xml",
        ],
        help="Optional list of package.xml paths.",
    )
    parser.add_argument(
        "--static_directive",
        type=str,
        default=None,
        help="Optional static directive for welded objects.",
    )
    
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_sdf_folder):
        os.makedirs(args.output_sdf_folder)

    # Load dataset
    hf_dataset, metadata = load_hf_dataset_with_metadata(
        args.scenes_data_path, 
    )
    hf_dataset.set_format(type="torch")

    # Apply Filter if specified
    current_scene_idx = args.scene_idx
    if args.filter_string:
        filtered_indices = []
        for idx, data in tqdm(
            enumerate(hf_dataset),
            desc="Searching scene with specified filter string",
            total=len(hf_dataset),
        ):
            if (
                "language_annotation" in data
                and args.filter_string in data["language_annotation"]
            ):
                filtered_indices.append(idx)
                if len(filtered_indices) > current_scene_idx:
                    break
        
        if not filtered_indices:
            raise ValueError(
                f"No scenes found with language_annotation containing '{args.filter_string}'"
            )
        current_scene_idx = filtered_indices[current_scene_idx]

    # Load and check Normalizer
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])
    if not normalizer.is_fitted:
        raise ValueError("Normalizer is not fitted!")

    # Create Scene Vector Description
    scene_vec_desc = get_scene_vec_description_from_metadata(
        metadata,
        static_directive=args.static_directive,
        package_names=args.package_names,
        package_file_paths=args.package_file_paths,
    )

    total_scenes = len(hf_dataset)
    print(f"Total scenes in dataset: {total_scenes}")

    # Determine indices to process (max 100 samples)
    if total_scenes <= 100:
        sample_indices = list(range(total_scenes))
    else:
        sample_indices = np.linspace(0, total_scenes - 1, num=100, dtype=int).tolist()

    # Processing Loop
    for idx in sample_indices:
        scene_data = hf_dataset[idx]
        scene_tensor = scene_data["scenes"][None]  # Shape (1, num_objects, num_features)

        # Un-normalize
        unnormalized_scene = normalizer.inverse_transform(
            scene_tensor.reshape(-1, scene_tensor.shape[-1])
        ).reshape(scene_tensor.shape)

        if "language_annotation" in scene_data:
            print("Language annotation:\n", scene_data["language_annotation"])

        assert len(unnormalized_scene.shape) == 3

        if isinstance(unnormalized_scene, torch.Tensor):
            unnormalized_scene = unnormalized_scene.cpu().detach().numpy()

        # Save individual instance SDFs
        for instance_i, scene_instance in enumerate(unnormalized_scene):
            sdf_out_path = f"{args.output_sdf_folder}/scene_{idx}_instance_{instance_i}.sdf"
            save_scene_sdf_for_blender_proc(
                sdf_out_path,
                scene_instance,
                scene_vec_desc,
            )


if __name__ == "__main__":
    main()