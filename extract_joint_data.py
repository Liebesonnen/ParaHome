import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path


# Helper function to make numpy arrays JSON serializable
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def extract_joint_data(scene_path, output_dir=None):
    """
    Extract joint position and state data from pickle files and save as JSON.

    Args:
        scene_path: Path to the scene directory containing pickle files
        output_dir: Directory to save JSON files (defaults to same as scene_path)
    """
    scene_path = Path(scene_path)
    if output_dir is None:
        output_dir = scene_path
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Process joint positions
    joint_positions_path = scene_path / "joint_positions.pkl"
    if joint_positions_path.exists():
        print(f"Processing {joint_positions_path}")
        with open(joint_positions_path, "rb") as f:
            joint_positions = pickle.load(f)

        # Convert to JSON serializable format
        json_positions = convert_to_serializable(joint_positions)

        # Save as JSON
        output_path = output_dir / "joint_positions.json"
        with open(output_path, "w") as f:
            json.dump(json_positions, f, indent=2)
        print(f"Saved joint positions to {output_path}")
        print(f"Joint positions shape: {np.array(joint_positions).shape}")
    else:
        print(f"Warning: {joint_positions_path} not found")

    # Process joint states
    joint_states_path = scene_path / "joint_states.pkl"
    if joint_states_path.exists():
        print(f"Processing {joint_states_path}")
        with open(joint_states_path, "rb") as f:
            joint_states = pickle.load(f)

        # Convert to JSON serializable format
        json_states = convert_to_serializable(joint_states)

        # Save as JSON
        output_path = output_dir / "joint_states.json"
        with open(output_path, "w") as f:
            json.dump(json_states, f, indent=2)
        print(f"Saved joint states to {output_path}")
        print(f"Joint states contains data for {len(joint_states)} frames")
    else:
        print(f"Warning: {joint_states_path} not found")


def main():
    parser = argparse.ArgumentParser(description="Extract joint data from pickle files and save as JSON")
    parser.add_argument("--scene_path", required=True, help="Path to the scene directory containing the pickle files")
    parser.add_argument("--output_dir", default=None, help="Directory to save the JSON files (defaults to scene path)")
    args = parser.parse_args()

    extract_joint_data(args.scene_path, args.output_dir)


if __name__ == "__main__":
    main()