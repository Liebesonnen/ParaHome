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


def analyze_all_scenes_for_objects(scenes_root, target_objects):
    """
    分析所有场景中与指定物体相关的动作

    Args:
        scenes_root: 场景根目录 (e.g., "data/seq")
        target_objects: 目标物体列表 (e.g., ["washingmachine", "chair", "drawer", ...])

    Returns:
        Dict containing statistics and action information
    """
    # 统计信息
    stats = {obj: 0 for obj in target_objects}
    action_details = {obj: [] for obj in target_objects}

    # 遍历所有场景
    processed_scenes = 0
    for scene_id in range(1, 208):  # s1 to s207
        scene_name = f"s{scene_id}"
        scene_path = os.path.join(scenes_root, scene_name)

        if not os.path.exists(scene_path):
            continue

        annotation_path = os.path.join(scene_path, "text_annotations.json")
        if not os.path.exists(annotation_path):
            print(f"Annotations not found for {scene_name}")
            continue

        processed_scenes += 1
        if processed_scenes % 20 == 0:
            print(f"Processed {processed_scenes} scenes...")

        # 读取标注文件
        try:
            with open(annotation_path, "r") as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"Error reading annotations for {scene_name}: {e}")
            continue

        # 分析每个动作
        for frame_range, annotation in annotations.items():
            annotation_lower = annotation.lower()

            # 检查是否与目标物体相关
            for obj in target_objects:
                obj_keywords = get_object_keywords(obj)

                if any(keyword in annotation_lower for keyword in obj_keywords):
                    stats[obj] += 1

                    # 解析帧范围
                    try:
                        start_frame, end_frame = map(int, frame_range.split())
                        action_details[obj].append({
                            "scene_id": scene_name,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "annotation": annotation,
                            "scene_path": scene_path
                        })
                    except ValueError:
                        print(f"Invalid frame range format: {frame_range} in {scene_name}")
                        continue

    print(f"Total processed scenes: {processed_scenes}")
    return stats, action_details


def get_object_keywords(object_name):
    """
    获取物体相关的关键词
    """
    keyword_mapping = {
        "washingmachine": ["washing machine", "washing", "washer", "laundry"],
        "chair": ["chair", "seat", "sit"],
        "drawer": ["drawer", "pull out", "open drawer", "close drawer", "draw"],
        "microwave": ["microwave", "micro wave", "heat", "warm"],
        "refrigerator": ["refrigerator", "fridge", "freezer", "cold"],
        "trashbin": ["trash", "garbage", "bin", "waste", "throw"]
    }

    return keyword_mapping.get(object_name, [object_name.lower()])


def get_object_scan_name_mapping():
    """
    获取对象名称映射，从描述名称到scan目录名称
    """
    return {
        "washingmachine": "washingmachine",
        "chair": "chair",
        "drawer": "drawer",
        "microwave": "microwave",
        "refrigerator": "refrigerator",
        "trashbin": "trashbin"
    }


def extract_articulated_point_clouds_batch(action_details, target_objects, output_dir):
    """
    批量提取所有相关动作的点云数据
    """
    scan_root = os.path.join(ROOT_REPOSITORY, "data/scan")
    object_mapping = get_object_scan_name_mapping()

    total_extractions = sum(len(action_details[obj]) for obj in target_objects)
    current_extraction = 0

    for obj_name in target_objects:
        if not action_details[obj_name]:
            print(f"No actions found for {obj_name}")
            continue

        print(f"\n=== Extracting point clouds for {obj_name} ({len(action_details[obj_name])} actions) ===")

        scan_obj_name = object_mapping.get(obj_name, obj_name)

        for action_info in action_details[obj_name]:
            current_extraction += 1
            scene_path = action_info["scene_path"]
            start_frame = action_info["start_frame"]
            end_frame = action_info["end_frame"]
            scene_id = action_info["scene_id"]
            annotation = action_info["annotation"]

            print(
                f"[{current_extraction}/{total_extractions}] Processing {scene_id}: {annotation[:50]}... (frames {start_frame}-{end_frame})")

            try:
                extract_single_action_point_clouds(
                    scene_path, start_frame, end_frame, scan_obj_name, scene_id, output_dir
                )
            except Exception as e:
                print(f"Error processing {scene_id} {obj_name}: {e}")
                continue


def extract_single_action_point_clouds(scene_path, start_frame, end_frame, object_name, scene_id, output_dir):
    """
    提取单个动作的点云数据
    """
    # 加载对象变换数据
    object_transform_path = os.path.join(scene_path, "object_transformations.pkl")
    if not os.path.exists(object_transform_path):
        print(f"Object transformations not found: {object_transform_path}")
        return

    with open(object_transform_path, "rb") as f:
        object_transforms = pickle.load(f)

    # 获取有效帧
    valid_frames = [f for f in range(start_frame, end_frame + 1) if f in object_transforms]
    if not valid_frames:
        print(f"No valid frames found for {scene_id} between {start_frame} and {end_frame}")
        return

    # 查找对象的所有部分
    object_parts = []
    for frame in valid_frames:
        for key in object_transforms[frame].keys():
            if key.startswith(object_name + "_"):
                if key not in object_parts:
                    object_parts.append(key)

    if not object_parts:
        print(f"Object '{object_name}' not found in {scene_id} for frames {start_frame}-{end_frame}")
        return

    scan_root = os.path.join(ROOT_REPOSITORY, "data/scan")

    # 为每个部分提取点云
    for part_full_name in object_parts:
        part_name = part_full_name.split("_")[1]

        # 加载网格
        mesh_path = os.path.join(scan_root, object_name, "simplified", part_name + ".obj")
        if not os.path.exists(mesh_path):
            print(f"Mesh not found: {mesh_path}")
            continue

        # 加载网格并采样点
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if len(mesh.vertices) == 0:
                print(f"Empty mesh: {mesh_path}")
                continue

            points = np.asarray(mesh.sample_points_uniformly(number_of_points=500).points)
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            continue

        # 收集变换矩阵
        part_frames = []
        part_transforms = []

        for frame in valid_frames:
            if part_full_name in object_transforms[frame]:
                part_frames.append(frame)
                part_transforms.append(object_transforms[frame][part_full_name])

        if not part_frames:
            continue

        # 应用变换
        transformed_points = []
        for transform in part_transforms:
            try:
                # 确保变换矩阵的格式正确
                transform = np.array(transform)
                if transform.shape != (4, 4):
                    print(f"Invalid transformation matrix shape: {transform.shape}")
                    continue

                rotation = transform[:3, :3]
                translation = transform[:3, 3]
                transformed = np.dot(points, rotation.T) + translation
                transformed_points.append(transformed)
            except Exception as e:
                print(f"Error applying transformation: {e}")
                continue

        if not transformed_points:
            continue

        # 保存点云数据
        point_cloud = np.array(transformed_points)  # shape: (n_frames, 500, 3)

        frame_range_str = f"{start_frame}_{end_frame}"
        output_file = os.path.join(output_dir, f"{scene_id}_{object_name}_{part_name}_{frame_range_str}.npy")

        np.save(output_file, point_cloud)
        print(f"Saved: {os.path.basename(output_file)}, shape: {point_cloud.shape}")


def main():
    parser = argparse.ArgumentParser(description="Batch extract articulated object point clouds from all scenes")
    parser.add_argument("--scenes_root", default="data/seq", help="Root directory containing all scenes")
    parser.add_argument("--output_dir", default="output_batch", help="Directory to save output files")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only generate statistics, don't extract point clouds")

    args = parser.parse_args()

    # 目标物体列表 (使用scan目录中的实际名称)
    target_objects = ["washingmachine", "chair", "drawer", "microwave", "refrigerator", "trashbin"]

    # 确保路径正确
    scenes_root = args.scenes_root
    if not os.path.isabs(scenes_root):
        scenes_root = os.path.join(ROOT_REPOSITORY, scenes_root)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(ROOT_REPOSITORY, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print("=== Analyzing all scenes for target objects ===")
    print(f"Target objects: {target_objects}")
    print(f"Scenes root: {scenes_root}")
    print(f"Output directory: {output_dir}")

    # 分析所有场景
    print("Scanning all scenes...")
    stats, action_details = analyze_all_scenes_for_objects(scenes_root, target_objects)

    # 保存统计信息
    stats_file = os.path.join(output_dir, "object_action_statistics.json")
    stats_data = {
        "statistics": stats,
        "total_actions": sum(stats.values()),
        "detailed_counts": {obj: len(action_details[obj]) for obj in target_objects},
        "action_details": action_details
    }

    with open(stats_file, "w") as f:
        json.dump(stats_data, f, indent=4)

    print(f"\n=== Statistics ===")
    print(f"Total actions found: {sum(stats.values())}")
    for obj, count in stats.items():
        print(f"{obj}: {count} actions")

    print(f"\nStatistics saved to: {stats_file}")

    if not args.stats_only:
        print(f"\n=== Extracting point clouds ===")
        extract_articulated_point_clouds_batch(action_details, target_objects, output_dir)
        print(f"\nPoint cloud extraction completed!")

        # 生成总结报告
        summary_file = os.path.join(output_dir, "extraction_summary.txt")
        with open(summary_file, "w") as f:
            f.write("ParaHome Articulated Objects Extraction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total scenes processed: s1 to s207\n")
            f.write(f"Target objects: {', '.join(target_objects)}\n")
            f.write(f"Total actions found: {sum(stats.values())}\n\n")

            f.write("Actions per object:\n")
            for obj, count in stats.items():
                f.write(f"  {obj}: {count} actions\n")

            f.write(f"\nPoint cloud files saved in: {output_dir}\n")
            f.write("File format: {scene_id}_{object_name}_{part_name}_{start_frame}_{end_frame}.npy\n")
            f.write("Data shape: (n_frames, 500, 3)\n")

        print(f"Summary saved to: {summary_file}")
    else:
        print("\nStatistics generation completed. Remove --stats_only flag to extract point clouds.")


if __name__ == "__main__":
    main()