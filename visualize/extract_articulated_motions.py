import os
import json
import pickle
import numpy as np
import argparse
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt

# Add parent directory to path
import sys

ROOT_REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_REPOSITORY)

SCAN_ROOT = os.path.join(ROOT_REPOSITORY, "data/scan")


def extract_articulated_object_motion(scene_root, action_name, start_frame, end_frame, output_dir, object_name=None,
                                      scene_id=None):
    """
    Extract and visualize the motion of articulated objects for a specific action.

    Args:
        scene_root: Path to the scene directory
        action_name: Name of the action (e.g., "Open cabinet")
        start_frame: Start frame of the action
        end_frame: End frame of the action
        output_dir: Directory to save the output files
        object_name: Optional manually specified object name
        scene_id: Scene identifier (e.g., "s1")
    """
    # Load object transformations
    object_transform_path = os.path.join(scene_root, "object_transformations.pkl")
    with open(object_transform_path, "rb") as f:
        object_transforms = pickle.load(f)

    # Load joint states to identify articulated parts
    joint_states_path = os.path.join(scene_root, "joint_states.pkl")
    with open(joint_states_path, "rb") as f:
        joint_states = pickle.load(f)

    # Load object in scene information
    object_in_scene_path = os.path.join(scene_root, "object_in_scene.json")
    with open(object_in_scene_path, "r") as f:
        objects_in_scene = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Frame range string for filenames
    frame_range_str = f"{start_frame}_{end_frame}"
    scene_prefix = f"{scene_id}_" if scene_id else ""

    articulated_objects = []

    # Use manually specified object if provided
    if object_name:
        # Find matching objects in the scene
        for scene_obj in objects_in_scene:
            if object_name.lower() in scene_obj.lower():
                articulated_objects.append(scene_obj)
                print(f"Found object matching '{object_name}': {scene_obj}")
    else:
        # Identify articulated objects from the action name
        action_lower = action_name.lower()

        # Keywords to identify articulated objects
        articulated_keywords = {
            "cabinet": ["cabinet"],
            "refrigerator": ["refrigerator", "fridge"],
            "microwave": ["microwave"],
            "pot": ["pot", "lid"],
            "drawer": ["drawer"],
            "washing machine": ["washing machine", "washer"],
            "gas stove": ["gas stove", "stove"]
        }

        # Find which articulated objects are involved in the action
        for obj, keywords in articulated_keywords.items():
            if any(keyword in action_lower for keyword in keywords):
                for scene_obj in objects_in_scene:
                    if obj in scene_obj:
                        articulated_objects.append(scene_obj)

    if not articulated_objects:
        print(f"No articulated objects found for action: {action_name}")
        if not object_name:
            print("Available objects in scene:")
            for obj in objects_in_scene:
                print(f"  - {obj}")
            print("Try specifying an object with --object_name")
        return

    print(f"Extracting motion for: {action_name} (Frames {start_frame}-{end_frame})")
    print(f"Articulated objects: {articulated_objects}")

    # Dictionary to store motion data
    motion_data = {}

    # Process each frame in the range
    for frame in range(start_frame, end_frame + 1):
        if frame not in object_transforms:
            continue

        for obj in articulated_objects:
            # Check both base and articulated parts
            for part in ["base", "part1", "part2"]:
                obj_part = f"{obj}_{part}"

                if obj_part in object_transforms[frame]:
                    if obj_part not in motion_data:
                        motion_data[obj_part] = []

                    # Get the transformation matrix
                    transform = object_transforms[frame][obj_part]
                    motion_data[obj_part].append(transform)

    # Save motion data as NPY files and visualize
    for obj_part, transforms in motion_data.items():
        if not transforms:
            continue

        # Convert to numpy array
        transforms_array = np.array(transforms)

        # Create filenames with scene_id and frame range
        filename_prefix = f"{scene_prefix}{action_name.replace(' ', '_')}_{frame_range_str}"

        # Save to NPY file
        output_file = os.path.join(output_dir, f"{filename_prefix}_{obj_part}_motion.npy")
        np.save(output_file, transforms_array)
        print(f"Saved motion data to: {output_file}")

        # Visualize the motion sequence
        visualize_motion_sequence(scene_root, obj_part, transforms, filename_prefix, output_dir)

        # Extract joint angles if available
        if obj_part.split('_')[0] in joint_states and frame in joint_states:
            joint_data = []
            for frame in range(start_frame, end_frame + 1):
                if frame in joint_states and obj_part.split('_')[0] in joint_states[frame]:
                    joint_data.append(joint_states[frame][obj_part.split('_')[0]])

            if joint_data:
                joint_array = np.array(joint_data)
                joint_file = os.path.join(output_dir, f"{filename_prefix}_{obj_part.split('_')[0]}_joints.npy")
                np.save(joint_file, joint_array)
                print(f"Saved joint data to: {joint_file}")

                # Plot joint angle changes
                plot_joint_changes(joint_array, obj_part.split('_')[0], f"{action_name} ({frame_range_str})",
                                   output_dir, filename_prefix)


def visualize_motion_sequence(scene_root, obj_part, transforms, filename_prefix, output_dir):
    """
    Visualize the motion sequence for an object part.

    Args:
        scene_root: Path to the scene directory
        obj_part: Object part name (e.g., "cabinet_base")
        transforms: List of transformation matrices
        filename_prefix: Prefix for output files (including scene_id and frame range)
        output_dir: Directory to save the output files
    """
    obj_name, part_name = obj_part.split('_')

    # Load the object mesh
    mesh_path = os.path.join(SCAN_ROOT, obj_name, "simplified", part_name + ".obj")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        return

    # Load mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Create a combined mesh showing the motion trajectory
    combined_mesh = o3d.geometry.TriangleMesh()

    # Sample points to avoid too dense visualization
    sample_rate = max(1, len(transforms) // 20)

    # Colors from blue to red to show motion sequence
    colors = plt.cm.plasma(np.linspace(0, 1, len(transforms[::sample_rate])))

    for i, transform in enumerate(transforms[::sample_rate]):
        # Create a copy of the mesh
        temp_mesh = o3d.geometry.TriangleMesh(mesh)

        # Apply transformation
        transform_matrix = np.array(transform)
        temp_mesh.transform(transform_matrix)

        # Color based on sequence
        color = colors[i][:3]  # RGB without alpha
        temp_mesh.paint_uniform_color(color)

        # Add to combined mesh
        combined_mesh += temp_mesh

    # Save the combined mesh
    output_file = os.path.join(output_dir, f"{filename_prefix}_{obj_part}_sequence.ply")
    o3d.io.write_triangle_mesh(output_file, combined_mesh)
    print(f"Saved visualization to: {output_file}")

    # Also create a point cloud for more efficient visualization
    pcd = combined_mesh.sample_points_uniformly(number_of_points=100000)
    pcd_file = os.path.join(output_dir, f"{filename_prefix}_{obj_part}_sequence_points.ply")
    o3d.io.write_point_cloud(pcd_file, pcd)
    print(f"Saved point cloud to: {pcd_file}")


def plot_joint_changes(joint_data, obj_name, action_title, output_dir, filename_prefix):
    """
    Plot changes in joint angles over time.

    Args:
        joint_data: Array of joint state values
        obj_name: Name of the object
        action_title: Title for the plot (action name with frame range)
        output_dir: Directory to save the output files
        filename_prefix: Prefix for output files
    """
    plt.figure(figsize=(10, 6))

    # For multi-dimensional joint data
    if joint_data.ndim > 1 and joint_data.shape[1] > 1:
        for i in range(joint_data.shape[1]):
            plt.plot(joint_data[:, i], label=f'Joint {i + 1}')
        plt.legend()
    else:
        plt.plot(joint_data)

    plt.title(f'Joint Motion for {obj_name} during "{action_title}"')
    plt.xlabel('Frame')
    plt.ylabel('Joint Value (radians/meters)')
    plt.grid(True)

    output_file = os.path.join(output_dir, f"{filename_prefix}_{obj_name}_joint_plot.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved joint motion plot to: {output_file}")


def parse_text_annotations(scene_root):
    """
    Parse the text annotations file to get action frame ranges.

    Args:
        scene_root: Path to the scene directory

    Returns:
        List of (action_name, start_frame, end_frame) tuples
    """
    # Try both possible filenames (with and without 's')
    possible_filenames = ["text_annotations.json", "text_annotation.json"]
    annotations_path = None

    for filename in possible_filenames:
        path = os.path.join(scene_root, filename)
        if os.path.exists(path):
            annotations_path = path
            break

    if annotations_path is None:
        raise FileNotFoundError(f"Could not find annotation file in {scene_root}. Tried: {possible_filenames}")

    print(f"Found annotation file: {annotations_path}")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    actions = []
    for frame_range, action_name in annotations.items():
        start_frame, end_frame = map(int, frame_range.split())
        actions.append((action_name, start_frame, end_frame))

    return actions


def get_scene_id(scene_path):
    """Extract scene ID from path (e.g., 's1' from 'data/seq/s1')"""
    return os.path.basename(scene_path)


def main():
    parser = argparse.ArgumentParser(description="Extract articulated object motions from ParaHome data")
    parser.add_argument("--scene_root", default="data/seq/s1", help="Path to the scene directory")
    parser.add_argument("--output_dir", default="motion_data", help="Directory to save output files")
    parser.add_argument("--action", default=None, help="Action name to search for (e.g., 'Open refrigerator')")
    parser.add_argument("--object_name", default=None, help="Manually specify an object name (e.g., 'refrigerator')")
    parser.add_argument("--all", action="store_true", help="Process all found actions without asking for selection")
    args = parser.parse_args()

    # Check if scene_root is relative or absolute
    if not os.path.isabs(args.scene_root):
        scene_root = os.path.join(ROOT_REPOSITORY, args.scene_root)
    else:
        scene_root = args.scene_root

    # Get scene ID for filenames
    scene_id = get_scene_id(args.scene_root)

    # Print directory information for debugging
    print(f"Looking for data in: {scene_root}")
    print(f"Scene ID: {scene_id}")
    print(f"Files in directory: {os.listdir(scene_root)}")

    # Create output directory
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(ROOT_REPOSITORY, args.output_dir)
    else:
        output_dir = args.output_dir

    # Parse text annotations
    all_actions = parse_text_annotations(scene_root)
    print(f"Found {len(all_actions)} actions in the annotation file")

    # Filter actions by the provided action string
    if args.action:
        matching_actions = []
        for action_name, start_frame, end_frame in all_actions:
            if args.action.lower() in action_name.lower():
                matching_actions.append((action_name, start_frame, end_frame))

        if not matching_actions:
            print(f"No actions found containing: '{args.action}'")
            print("Available actions:")
            for action_name, start_frame, end_frame in all_actions:
                print(f"  - {action_name} (Frames {start_frame}-{end_frame})")
            return

        print(f"Found {len(matching_actions)} actions matching '{args.action}':")
        for i, (action_name, start_frame, end_frame) in enumerate(matching_actions):
            print(f"  {i + 1}. {action_name} (Frames {start_frame}-{end_frame})")

        # Process all matching actions or ask for selection
        if args.all:
            actions_to_process = matching_actions
            print("Processing all matching actions...")
        else:
            # Ask user which action to process
            if len(matching_actions) == 1:
                # If there's only one match, use it automatically
                selection = 1
                actions_to_process = [matching_actions[0]]
                print(f"Using the only matching action: {matching_actions[0][0]}")
            else:
                selection = input("\nSelect action number to process (or 'all' for all): ")

                if selection.lower() == 'all':
                    actions_to_process = matching_actions
                else:
                    try:
                        selection = int(selection)
                        if 1 <= selection <= len(matching_actions):
                            actions_to_process = [matching_actions[selection - 1]]
                        else:
                            print(f"Invalid selection. Please choose 1-{len(matching_actions)}")
                            return
                    except ValueError:
                        print("Invalid input. Please enter a number or 'all'")
                        return
    else:
        # No action specified, list all actions
        print("Available actions:")
        for i, (action_name, start_frame, end_frame) in enumerate(all_actions):
            print(f"  {i + 1}. {action_name} (Frames {start_frame}-{end_frame})")

        if not args.object_name:
            print("\nPlease specify an action with --action or an object with --object_name")
            return

        # If object name is specified without action, process all actions
        print(f"\nProcessing all actions with manually specified object: {args.object_name}")
        actions_to_process = all_actions

    # Process the selected actions
    for action_name, start_frame, end_frame in actions_to_process:
        extract_articulated_object_motion(
            scene_root,
            action_name,
            start_frame,
            end_frame,
            output_dir,
            object_name=args.object_name,
            scene_id=scene_id
        )


if __name__ == "__main__":
    main()
