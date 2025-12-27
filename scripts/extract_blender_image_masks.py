"""
Script to extract images, masks, and metadata from BlenderProc HDF5 output.
Usage:
    # Single file
    python extract_blender_image_masks.py --hdf5_path ./render_output/scene_0.hdf5

    # Walk a directory recursively and extract all .hdf5 files in parallel
    python extract_blender_image_masks.py --input_dir ./render_output --n_jobs -1
"""

import argparse
import json
import logging
import os
from typing import Dict, Any, List, Optional

from joblib import Parallel, delayed

import h5py
import numpy as np
import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger(__name__)


class HDF5Extractor:
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found at: {hdf5_path}")

    def load_data(self) -> Dict[str, Any]:
        """
        Loads necessary data from the HDF5 file similar to a Dataset __getitem__.
        """
        console_logger.info(f"Loading data from {self.hdf5_path}...")
        
        with h5py.File(self.hdf5_path, "r") as f:
            # Load raw data
            # Note: BlenderProc usually saves 'colors' as list of images if multiple frames
            # Adjusting indexing [0] assuming single frame or grabbing first frame for simplicity
            # If your HDF5 has shape (H,W,3) directly, remove the [0] index.
            if f["colors"].ndim == 4: # (Frames, H, W, 3)
                color_image = f["colors"][0] 
                mask_data = f["instance_segmaps"][0]
                all_points = f["point_clouds"][0]
            else: # (H, W, 3)
                color_image = f["colors"][:]
                mask_data = f["instance_segmaps"][:]
                all_points = f["point_clouds"][:]

            camera_K = np.array(f["camera_intrinsics"])
            
            # Parse attributes
            attr_map_raw = f["instance_attribute_maps"][()]
            if isinstance(attr_map_raw, bytes):
                attr_map_raw = attr_map_raw.decode("utf-8")
            parsed_attributes = json.loads(attr_map_raw)

        # Process Objects
        # Count valid instances (those with a mesh file)
        valid_instances = [inst for inst in parsed_attributes if inst.get("mesh_file") is not None]
        N = len(valid_instances)
        H, W = color_image.shape[:2]

        object_points = torch.zeros((N, H, W, 3), dtype=torch.float32)
        object_masks = torch.zeros((N, H, W), dtype=torch.float32)
        instance_ids = torch.zeros(N, dtype=torch.long)
        instance_mesh_files = []

        for obj_idx, inst in enumerate(valid_instances):
            inst_idx = inst["idx"]
            mesh_rel_path = inst["mesh_file"]
            
            # Create Binary Mask
            # Note: mask_data contains instance IDs
            obj_mask = (mask_data == inst_idx).astype(np.float32)
            
            # Store data
            object_masks[obj_idx] = torch.from_numpy(obj_mask)
            instance_ids[obj_idx] = inst_idx
            instance_mesh_files.append(mesh_rel_path)

            # Mask the point cloud for this specific object
            # (H, W, 3) * (H, W, 1) -> (H, W, 3)
            obj_points_masked = all_points * obj_mask[..., None]
            object_points[obj_idx] = torch.from_numpy(obj_points_masked)

        # Prepare final dictionary
        # Normalize image to 0-1 float for consistency with your snippet
        image_tensor = torch.from_numpy(color_image).float().permute(2, 0, 1) / 255.0
        intrinsics = torch.from_numpy(camera_K).float()

        return {
            "hdf5_path": self.hdf5_path,
            "image": image_tensor,           # (3, H, W)
            "intrinsics": intrinsics,        # (3, 3)
            "object_points": object_points,  # (N, H, W, 3)
            "object_masks": object_masks,    # (N, H, W)
            "instance_ids": instance_ids,    # (N,)
            "mesh_files": instance_mesh_files, # List[str]
        }

    def save_images_and_masks(self, data: Dict[str, Any]):
        """
        Extracts images and masks from loaded data and saves them to disk.
        """
        hdf5_path = data['hdf5_path']
        hdf5_dir = os.path.dirname(hdf5_path)
        hdf_name = os.path.basename(hdf5_path)
        output_dir = os.path.join(hdf5_dir, f"{os.path.splitext(hdf_name)[0]}_output")
        
        os.makedirs(output_dir, exist_ok=True)
        console_logger.info(f"Saving output to: {output_dir}")

        # 1. Save RGB Image
        # Convert (3, H, W) -> (H, W, 3) -> uint8
        image_np = (data['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(image_np).save(os.path.join(output_dir, "image.png"))

        # 2. Save Per-Object Masked Images (RGBA)
        N = data['object_masks'].shape[0]
        mesh_files = data['mesh_files']

        for i in range(N):
            # Convert Mask to uint8 (0 or 255)
            mask_np = (data['object_masks'][i].numpy() * 255).astype(np.uint8)
            
            # Create RGBA Image
            rgba_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
            rgba_image[..., :3] = image_np  # RGB channels
            rgba_image[..., 3] = mask_np    # Alpha channel

            # Save
            # Using index 'i' as filename, or you could use instance ID
            out_name = f"{i}.png"
            Image.fromarray(rgba_image).save(os.path.join(output_dir, out_name))

        # 3. Save Metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        metadata = {
            "origin_hdf5": hdf5_path,
            "num_objects": N,
            "mesh_files": mesh_files
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        console_logger.info("Extraction complete.")


def _output_dir_for_hdf5(hdf5_path: str) -> str:
    hdf5_dir = os.path.dirname(hdf5_path)
    hdf_name = os.path.basename(hdf5_path)
    return os.path.join(hdf5_dir, f"{os.path.splitext(hdf_name)[0]}_output")


def find_hdf5_files(input_dir: str) -> List[str]:
    """
    Recursively walk a directory and return all *.hdf5 file paths (sorted).
    """
    hdf5_files: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, fname))
    hdf5_files.sort()
    return hdf5_files


def extract_one_hdf5(hdf5_path: str, overwrite: bool = False) -> Optional[str]:
    """
    Extract a single HDF5 file.

    Returns:
        - None if extraction ran successfully
        - A short string reason if the file was skipped
    """
    output_dir = _output_dir_for_hdf5(hdf5_path)
    metadata_path = os.path.join(output_dir, "metadata.json")

    if not overwrite and os.path.exists(metadata_path):
        console_logger.info(f"Skipping (already extracted): {hdf5_path}")
        return "already_extracted"

    extractor = HDF5Extractor(hdf5_path)
    data = extractor.load_data()
    extractor.save_images_and_masks(data)
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract images and masks from HDF5.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hdf5_path", type=str, help="Path to a single source HDF5 file.")
    group.add_argument("--input_dir", type=str, help="Directory to walk recursively for *.hdf5 files.")

    parser.add_argument("--n_jobs", type=int, default=-1, help="Joblib parallelism (default: -1 = all cores).")
    parser.add_argument(
        "--backend",
        type=str,
        default="loky",
        choices=["loky", "multiprocessing", "threading"],
        help="Joblib backend (default: loky).",
    )
    parser.add_argument("--joblib_verbose", type=int, default=0, help="Joblib verbosity (default: 0).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist.")
    parser.add_argument("--dry_run", action="store_true", help="Only list discovered HDF5 files; do not extract.")
    args = parser.parse_args()

    if args.hdf5_path:
        try:
            extract_one_hdf5(args.hdf5_path, overwrite=args.overwrite)
        except Exception as e:
            console_logger.error(f"Failed to extract HDF5 ({args.hdf5_path}): {e}")
            raise
        return

    # Directory mode
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"--input_dir must be a directory. Got: {input_dir}")

    hdf5_files = find_hdf5_files(input_dir)
    console_logger.info(f"Discovered {len(hdf5_files)} .hdf5 files under: {input_dir}")

    if args.dry_run:
        for p in hdf5_files:
            console_logger.info(p)
        return

    def _safe_extract(path: str) -> Optional[str]:
        try:
            return extract_one_hdf5(path, overwrite=args.overwrite)
        except Exception as e:
            # Include the path in the exception message so joblib surfaces context.
            raise RuntimeError(f"Failed to extract {path}: {e}") from e

    Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=args.joblib_verbose)(
        delayed(_safe_extract)(p) for p in hdf5_files
    )

if __name__ == "__main__":
    main()
