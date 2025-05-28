import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
import open3d as o3d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Define root repository path
ROOT_REPOSITORY = os.path.dirname(os.path.abspath(__file__))


def find_action_frame_ranges(scene_path, action_name):
    """
    Find all frame ranges for a specific action in the text_annotations.json file.

    Args:
        scene_path: Path to the scene directory (e.g., "data/seq/s1")
        action_name: Name of the action to find (e.g., "Open refrigerator")

    Returns:
        List of tuples (start_frame, end_frame) for all occurrences of the action
    """
    # Get the scene ID from the path (e.g., "s1" from "data/seq/s1")
    scene_id = os.path.basename(scene_path)

    # Load text annotations
    annotation_path = os.path.join(scene_path, "text_annotation.json")
    if not os.path.exists(annotation_path):
        print(f"Error: Annotations file not found at {annotation_path}")
        return []

    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    # Find all frame ranges for the action
    frame_ranges = []
    for frame_range, annotation in annotations.items():
        if action_name.lower() in annotation.lower():
            # Frame range is in format "start_frame end_frame"
            start_frame, end_frame = map(int, frame_range.split())
            frame_ranges.append((start_frame, end_frame, annotation))

    if not frame_ranges:
        print(f"No frame ranges found for action '{action_name}'")
    else:
        print(f"Found {len(frame_ranges)} frame ranges for action '{action_name}':")
        for i, (start, end, ann) in enumerate(frame_ranges):
            print(f"  {i + 1}. Frames {start}-{end}: {ann}")

    return frame_ranges


def extract_articulated_point_clouds(scene_path, start_frame, end_frame, object_name, output_dir="output",
                                     resample_to=None):
    """
    Extract point clouds of articulated objects for specific frames.

    Args:
        scene_path: Path to the scene directory (e.g., "data/seq/s1")
        start_frame: Start frame number
        end_frame: End frame number
        object_name: Name of the object to extract (e.g., "microwave")
        output_dir: Directory to save output files
        resample_to: Optional number of frames to resample to (if None, keep original number)
    """
    # Get the scene ID from the path (e.g., "s1" from "data/seq/s1")
    scene_id = os.path.basename(scene_path)

    # Make sure paths are absolute
    scene_root = scene_path
    if not os.path.isabs(scene_root):
        scene_root = os.path.join(ROOT_REPOSITORY, scene_path)

    scan_root = os.path.join(ROOT_REPOSITORY, "data/scan")
    os.makedirs(output_dir, exist_ok=True)

    frame_range_str = f"{start_frame}_{end_frame}"
    print(f"Extracting {object_name} from {scene_root} frames {start_frame}-{end_frame}")

    # Load object transformations
    object_transform_path = os.path.join(scene_root, "object_transformations.pkl")
    with open(object_transform_path, "rb") as f:
        object_transforms = pickle.load(f)

    # Load joint states
    joint_states_path = os.path.join(scene_root, "joint_states.pkl")
    with open(joint_states_path, "rb") as f:
        joint_states = pickle.load(f)

    # Get valid frames within the range
    valid_frames = [f for f in range(start_frame, end_frame + 1) if f in object_transforms]
    if not valid_frames:
        print(f"No valid frames found between {start_frame} and {end_frame}")
        return

    # Find all parts of the specified object
    object_parts = []
    for frame in valid_frames:
        for key in object_transforms[frame].keys():
            if key.startswith(object_name + "_"):
                part_full_name = key
                if part_full_name not in object_parts:
                    object_parts.append(part_full_name)

    if not object_parts:
        print(f"Object '{object_name}' not found in the scene for the specified frames")
        return

    print(f"Found parts for {object_name}: {object_parts}")

    # Extract point clouds for each part
    for part_full_name in object_parts:
        part_name = part_full_name.split("_")[1]

        # Load the object mesh
        mesh_path = os.path.join(scan_root, object_name, "simplified", part_name + ".obj")
        if not os.path.exists(mesh_path):
            print(f"Mesh not found: {mesh_path}")
            continue

        print(f"Loading mesh: {mesh_path}")

        # Load mesh and sample points
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        points = np.asarray(mesh.sample_points_uniformly(number_of_points=500).points)

        # Apply transformations for each frame
        part_frames = []
        part_transforms = []

        for frame in valid_frames:
            if part_full_name in object_transforms[frame]:
                part_frames.append(frame)
                part_transforms.append(object_transforms[frame][part_full_name])

        if not part_frames:
            print(f"No transformations found for {part_full_name}")
            continue

        print(f"Processing {len(part_frames)} frames for {part_full_name}")

        # Transform points for each frame
        transformed_points = []
        for transform in part_transforms:
            # Extract rotation and translation
            rotation = transform[:3, :3]
            translation = transform[:3, 3]

            # Apply transformation to all points
            transformed = np.dot(points, rotation.T) + translation
            transformed_points.append(transformed)

        # Resample if requested
        if resample_to is not None and len(transformed_points) != resample_to:
            print(f"Resampling {len(transformed_points)} frames to {resample_to} frames for {part_full_name}")
            transformed_points = resample_point_cloud_sequence(transformed_points, resample_to)

        # Convert to numpy array and save
        point_cloud = np.array(transformed_points)
        output_file = os.path.join(output_dir, f"{scene_id}_{object_name}_{part_name}_{frame_range_str}.npy")
        np.save(output_file, point_cloud)
        print(f"Saved point cloud to: {output_file}, shape: {point_cloud.shape}")

    # Extract joint parameters
    extract_joint_parameters(scene_id, object_name, valid_frames, joint_states, output_dir, frame_range_str,
                             resample_to)


def resample_point_cloud_sequence(point_clouds, num_frames):
    """
    Resample a sequence of point clouds to a specific number of frames.

    Args:
        point_clouds: List of point cloud arrays
        num_frames: Target number of frames

    Returns:
        List of resampled point clouds
    """
    if len(point_clouds) == num_frames:
        return point_clouds

    # Create a time axis for the original sequence
    original_times = np.linspace(0, 1, len(point_clouds))
    # Create a time axis for the target sequence
    target_times = np.linspace(0, 1, num_frames)

    # Reshape point clouds to (n_frames, n_points * 3)
    n_points = point_clouds[0].shape[0]
    flattened = np.array([pc.flatten() for pc in point_clouds])

    # Interpolate each coordinate
    interpolator = interp1d(original_times, flattened, axis=0, kind='linear')
    resampled_flat = interpolator(target_times)

    # Reshape back to (n_frames, n_points, 3)
    resampled = [arr.reshape(n_points, 3) for arr in resampled_flat]

    return resampled


def extract_joint_parameters(scene_id, object_name, frames, joint_states, output_dir, frame_range_str,
                             resample_to=None):
    """
    Extract joint parameters for an articulated object.

    Args:
        scene_id: ID of the scene (e.g., "s1")
        object_name: Name of the object
        frames: List of frame numbers
        joint_states: Dictionary of joint states
        output_dir: Directory to save output files
        frame_range_str: String representation of frame range for filename
        resample_to: Optional number of frames to resample to
    """
    # Extract joint parameters for all frames
    joint_data = []

    # Check if the object has joint information
    has_joint_info = False
    for frame in frames:
        if frame in joint_states and object_name in joint_states[frame]:
            has_joint_info = True
            break

    if not has_joint_info:
        print(f"No joint information found for {object_name}")
        return

    print(f"Extracting joint parameters for {object_name}")

    # Extract joint states for each frame
    for frame in frames:
        if frame in joint_states and object_name in joint_states[frame]:
            joint_data.append(joint_states[frame][object_name])

    # Resample if requested
    if resample_to is not None and len(joint_data) != resample_to:
        print(f"Resampling {len(joint_data)} joint frames to {resample_to} frames")
        joint_data = resample_joint_data(joint_data, resample_to)

    # Save joint data
    joint_data = np.array(joint_data)
    joint_file = os.path.join(output_dir, f"{scene_id}_{object_name}_joints_{frame_range_str}.npy")
    np.save(joint_file, joint_data)
    print(f"Saved joint states to: {joint_file}, shape: {joint_data.shape}")

    # Plot joint motion
    plot_joint_changes(joint_data, object_name, scene_id, frame_range_str, output_dir)

    # Try to extract joint axis information
    try:
        joint_info_path = os.path.join(ROOT_REPOSITORY, "data", "joint_info")
        if os.path.exists(joint_info_path):
            print(f"Loading joint info from: {joint_info_path}")
            with open(joint_info_path, "rb") as f:
                joint_info = pickle.load(f)

            # Save joint metadata if it exists for this object
            if isinstance(joint_info, dict) and object_name in joint_info:
                joint_meta_file = os.path.join(output_dir,
                                               f"{scene_id}_{object_name}_joint_metadata_{frame_range_str}.json")
                with open(joint_meta_file, "w") as f:
                    json.dump(joint_info[object_name], f, indent=4)
                print(f"Saved joint metadata to: {joint_meta_file}")
    except Exception as e:
        print(f"Error extracting joint metadata: {e}")
        print("Joint metadata could not be extracted. Only joint states are available.")


def resample_joint_data(joint_data, num_frames):
    """
    Resample joint data to a specific number of frames.

    Args:
        joint_data: List of joint state arrays
        num_frames: Target number of frames

    Returns:
        List of resampled joint states
    """
    # Convert to numpy array
    joint_data = np.array(joint_data)

    # Create a time axis for the original sequence
    original_times = np.linspace(0, 1, len(joint_data))
    # Create a time axis for the target sequence
    target_times = np.linspace(0, 1, num_frames)

    # Interpolate each joint parameter
    if joint_data.ndim == 1:
        # Single joint parameter
        interpolator = interp1d(original_times, joint_data, axis=0, kind='linear')
        resampled = interpolator(target_times)
    else:
        # Multiple joint parameters
        interpolator = interp1d(original_times, joint_data, axis=0, kind='linear')
        resampled = interpolator(target_times)

    return resampled


def plot_joint_changes(joint_data, object_name, scene_id, frame_range_str, output_dir):
    """
    Plot changes in joint angles over time.

    Args:
        joint_data: Array of joint state values
        object_name: Name of the object
        scene_id: ID of the scene
        frame_range_str: String representation of frame range
        output_dir: Directory to save the output files
    """
    plt.figure(figsize=(10, 6))

    # For multi-dimensional joint data
    if joint_data.ndim > 1 and joint_data.shape[1] > 1:
        for i in range(joint_data.shape[1]):
            plt.plot(joint_data[:, i], label=f'Joint {i + 1}')
        plt.legend()
    else:
        plt.plot(joint_data)

    plt.title(f'Joint Motion for {object_name} ({scene_id}, frames {frame_range_str.replace("_", "-")})')
    plt.xlabel('Frame')
    plt.ylabel('Joint Value (radians/meters)')
    plt.grid(True)

    output_file = os.path.join(output_dir, f"{scene_id}_{object_name}_joint_plot_{frame_range_str}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved joint motion plot to: {output_file}")


def visualize_point_cloud_sequence(point_cloud_file):
    """
    Visualize a sequence of point clouds (optional visualization).

    Args:
        point_cloud_file: Path to the point cloud npy file
    """
    # Load point cloud
    point_clouds = np.load(point_cloud_file)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    # Add coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coord)

    # Create point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set first frame
    pcd.points = o3d.utility.Vector3dVector(point_clouds[0])
    vis.add_geometry(pcd)

    print(f"Visualizing {len(point_clouds)} frames. Press 'q' to exit.")

    # Show animation
    for i in range(1, len(point_clouds)):
        pcd.points = o3d.utility.Vector3dVector(point_clouds[i])
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Small delay for smoother visualization
        import time
        time.sleep(0.05)

    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract articulated object point clouds and joint parameters")
    parser.add_argument("scene_path", help="Path to the scene directory (e.g., 'data/seq/s1')")
    parser.add_argument("action", help="Action name to extract (e.g., 'Open refrigerator')")
    parser.add_argument("--object_name", help="Name of the object to extract (if not automatically detected)")
    parser.add_argument("--output_dir", default="output", help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", help="Visualize the extracted point clouds")
    parser.add_argument("--resample", type=int, help="Resample to specified number of frames (default: keep original)")

    args = parser.parse_args()

    # Find frame ranges for the specified action
    frame_ranges = find_action_frame_ranges(args.scene_path, args.action)

    if not frame_ranges:
        sys.exit(1)

    # Ask the user to select a frame range
    if len(frame_ranges) == 1:
        selected_range = 0
    else:
        while True:
            try:
                choice = int(input(f"Select a frame range (1-{len(frame_ranges)}): "))
                if 1 <= choice <= len(frame_ranges):
                    selected_range = choice - 1
                    break
                else:
                    print(f"Please enter a number between 1 and {len(frame_ranges)}")
            except ValueError:
                print("Please enter a valid number")

    # Get the selected frame range
    start_frame, end_frame, annotation = frame_ranges[selected_range]
    print(f"Selected frame range: {start_frame}-{end_frame} ({annotation})")

    # Try to extract object name from the action if not provided
    object_name = args.object_name
    if not object_name:
        # Common objects to look for in the action name
        objects = ["microwave", "refrigerator", "cabinet", "drawer", "gasstove",
                   "sink", "washing", "pot", "kettle", "laptop", "glasses", "chair"]

        for obj in objects:
            if obj.lower() in args.action.lower() or obj.lower() in annotation.lower():
                object_name = obj
                break

    if not object_name:
        object_name = input("Could not detect object name. Please enter the object name: ")

    print(f"Using object name: {object_name}")

    # Extract point clouds and joint parameters
    extract_articulated_point_clouds(
        args.scene_path,
        start_frame,
        end_frame,
        object_name,
        args.output_dir,
        args.resample
    )

    # Optionally visualize the results
    if args.visualize:
        scene_id = os.path.basename(args.scene_path)
        frame_range_str = f"{start_frame}_{end_frame}"

        for file in os.listdir(args.output_dir):
            if file.endswith(
                    ".npy") and scene_id in file and object_name in file and frame_range_str in file and not "joints" in file:
                file_path = os.path.join(args.output_dir, file)
                print(f"Visualizing: {file}")
                visualize_point_cloud_sequence(file_path)