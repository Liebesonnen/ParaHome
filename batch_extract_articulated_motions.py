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
from collections import defaultdict

# Define root repository path
ROOT_REPOSITORY = os.path.dirname(os.path.abspath(__file__))
SCAN_ROOT = os.path.join(ROOT_REPOSITORY, "data/scan")

# 目标物体类型
TARGET_OBJECTS = ['washing machine', 'chair', 'drawer', 'microwave', 'refrigerator', 'trashbin']


def get_object_keywords():
    """Define keywords for each target object type"""
    return {
        'washing machine': ['washing', 'washer', 'machine'],
        'chair': ['chair', 'seat'],
        'drawer': ['drawer'],
        'microwave': ['microwave', 'oven'],
        'refrigerator': ['refrigerator', 'fridge'],
        'trashbin': ['trash', 'bin', 'garbage']
    }


def identify_object_type_from_name(obj_name, object_keywords):
    """Identify which target object type an object belongs to"""
    obj_name_lower = obj_name.lower()
    for obj_type, keywords in object_keywords.items():
        if any(keyword in obj_name_lower for keyword in keywords):
            return obj_type
    return None


def identify_object_type_from_action(action_name, object_keywords):
    """Identify which target object type is involved in an action"""
    action_lower = action_name.lower()
    for obj_type, keywords in object_keywords.items():
        if any(keyword in action_lower for keyword in keywords):
            return obj_type
    return None


def analyze_all_scenes_for_objects(scenes_root, target_objects, object_keywords):
    """
    分析所有场景中与指定物体相关的动作
    """
    # 统计信息
    stats = initialize_stats()

    # 遍历所有场景
    processed_scenes = 0
    for scene_id in range(1, 208):  # s1 to s207
        scene_name = f"s{scene_id}"
        scene_path = os.path.join(scenes_root, scene_name)

        if not os.path.exists(scene_path):
            continue

        processed_scenes += 1
        if processed_scenes % 20 == 0:
            print(f"Analyzing scene {scene_name}... ({processed_scenes}/207)")

        # 解析动作标注
        actions = parse_text_annotations(scene_path)
        if not actions:
            continue

        # 加载场景中的物体信息
        object_in_scene_path = os.path.join(scene_path, "object_in_scene.json")
        if not os.path.exists(object_in_scene_path):
            continue

        with open(object_in_scene_path, "r") as f:
            objects_in_scene = json.load(f)

        # 分析每个动作
        for action_name, start_frame, end_frame in actions:
            # 识别动作中涉及的目标物体类型
            target_object_type = identify_object_type_from_action(action_name, object_keywords)

            # 如果动作名称中没有找到目标物体，跳过
            if not target_object_type:
                continue

            # 找到场景中属于该类型的所有物体
            articulated_objects = []
            for scene_obj in objects_in_scene:
                obj_type = identify_object_type_from_name(scene_obj, object_keywords)
                if obj_type == target_object_type:
                    articulated_objects.append(scene_obj)

            if not articulated_objects:
                continue

            # 更新统计信息
            stats[target_object_type]['total_actions'] += 1
            stats[target_object_type]['scenes'].add(scene_name)
            stats[target_object_type]['actions'].append({
                'scene_id': scene_name,
                'action_name': action_name,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'objects': articulated_objects,
                'scene_path': scene_path
            })

    print(f"Total processed scenes: {processed_scenes}")
    return stats


def parse_text_annotations(scene_root):
    """Parse the text annotations file to get action frame ranges."""
    # Try both possible filenames
    possible_filenames = ["text_annotations.json", "text_annotation.json"]
    annotations_path = None

    for filename in possible_filenames:
        path = os.path.join(scene_root, filename)
        if os.path.exists(path):
            annotations_path = path
            break

    if annotations_path is None:
        return []

    try:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        actions = []
        for frame_range, action_name in annotations.items():
            try:
                start_frame, end_frame = map(int, frame_range.split())
                actions.append((action_name, start_frame, end_frame))
            except ValueError:
                continue

        return actions
    except Exception as e:
        return []


def initialize_stats():
    """Initialize statistics dictionary"""
    stats = {}
    for obj_type in TARGET_OBJECTS:
        stats[obj_type] = {
            'total_actions': 0,
            'scenes': set(),  # Will be converted to list later
            'actions': []
        }
    return stats


def extract_articulated_point_clouds_batch(stats, output_dir):
    """
    批量提取所有相关动作的点云数据
    """
    total_extractions = sum(len(stats[obj]['actions']) for obj in TARGET_OBJECTS)
    current_extraction = 0

    for obj_type in TARGET_OBJECTS:
        if not stats[obj_type]['actions']:
            print(f"No actions found for {obj_type}")
            continue

        print(f"\n=== Extracting point clouds for {obj_type} ({len(stats[obj_type]['actions'])} actions) ===")

        for action_info in stats[obj_type]['actions']:
            current_extraction += 1
            scene_path = action_info["scene_path"]
            start_frame = action_info["start_frame"]
            end_frame = action_info["end_frame"]
            scene_id = action_info["scene_id"]
            objects = action_info["objects"]
            action_name = action_info["action_name"]

            print(
                f"[{current_extraction}/{total_extractions}] Processing {scene_id}: {action_name[:50]}... (frames {start_frame}-{end_frame})")

            try:
                extract_single_action_point_clouds(
                    scene_path, start_frame, end_frame, objects, scene_id, action_name, output_dir
                )
            except Exception as e:
                print(f"Error processing {scene_id} {obj_type}: {e}")
                continue


def extract_single_action_point_clouds(scene_path, start_frame, end_frame, objects, scene_id, action_name, output_dir):
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

    # 为每个物体的每个部分提取点云
    for obj_name in objects:
        # 查找对象的所有部分
        object_parts = []
        for frame in valid_frames:
            for key in object_transforms[frame].keys():
                if key.startswith(obj_name + "_"):
                    if key not in object_parts:
                        object_parts.append(key)

        if not object_parts:
            print(f"Object '{obj_name}' not found in {scene_id} for frames {start_frame}-{end_frame}")
            continue

        # 为每个部分提取点云
        for part_full_name in object_parts:
            part_name = part_full_name.split("_")[1]

            # 加载网格
            mesh_path = os.path.join(SCAN_ROOT, obj_name, "simplified", part_name + ".obj")
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
            output_file = os.path.join(output_dir, f"{scene_id}_{obj_name}_{part_name}_{frame_range_str}.npy")

            np.save(output_file, point_cloud)
            print(f"Saved: {os.path.basename(output_file)}, shape: {point_cloud.shape}")


def save_stats(stats, output_dir):
    """Save statistics to JSON file"""
    # Convert sets to lists for JSON serialization
    stats_for_json = {}
    for obj_type, data in stats.items():
        stats_for_json[obj_type] = {
            'total_actions': data['total_actions'],
            'unique_scenes': list(data['scenes']),
            'scene_count': len(data['scenes']),
            'actions': data['actions']
        }

    # Add summary
    summary = {
        'total_target_actions': sum(data['total_actions'] for data in stats.values()),
        'total_unique_scenes_with_target_actions': len(set().union(*[data['scenes'] for data in stats.values()])),
        'object_types_processed': TARGET_OBJECTS
    }

    result = {
        'summary': summary,
        'by_object_type': stats_for_json
    }

    stats_file = os.path.join(output_dir, "target_object_actions_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nStatistics saved to: {stats_file}")
    return result


def print_summary(stats):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    total_actions = 0
    all_scenes = set()

    for obj_type in TARGET_OBJECTS:
        data = stats[obj_type]
        total_actions += data['total_actions']
        all_scenes.update(data['scenes'])

        print(f"\n{obj_type.upper()}:")
        print(f"  Total actions: {data['total_actions']}")
        print(f"  Scenes involved: {len(data['scenes'])}")
        if data['scenes']:
            scenes_list = sorted(data['scenes'], key=lambda x: int(x[1:]))  # Sort by scene number
            print(f"  Scene IDs: {', '.join(scenes_list[:10])}{' ...' if len(scenes_list) > 10 else ''}")

    print(f"\nOVERALL:")
    print(f"  Total target object actions: {total_actions}")
    print(f"  Total scenes with target actions: {len(all_scenes)}")
    print(f"  Scene coverage: {len(all_scenes)}/207 ({len(all_scenes) / 207 * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Batch extract articulated object point clouds from all scenes")
    parser.add_argument("--scenes_root", default="data/seq", help="Root directory containing all scenes")
    parser.add_argument("--output_dir", default="output_batch", help="Directory to save output files")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only generate statistics, don't extract point clouds")

    args = parser.parse_args()

    # 目标物体列表
    target_objects = TARGET_OBJECTS
    object_keywords = get_object_keywords()

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
    stats = analyze_all_scenes_for_objects(scenes_root, target_objects, object_keywords)

    # 保存统计信息
    save_stats(stats, output_dir)
    print_summary(stats)

    if not args.stats_only:
        print(f"\n=== Extracting point clouds ===")
        extract_articulated_point_clouds_batch(stats, output_dir)
        print(f"\nPoint cloud extraction completed!")

        # 生成总结报告
        summary_file = os.path.join(output_dir, "extraction_summary.txt")
        with open(summary_file, "w") as f:
            f.write("ParaHome Articulated Objects Extraction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total scenes processed: s1 to s207\n")
            f.write(f"Target objects: {', '.join(target_objects)}\n")
            f.write(f"Total actions found: {sum(stats[obj]['total_actions'] for obj in target_objects)}\n\n")

            f.write("Actions per object:\n")
            for obj_type in target_objects:
                f.write(f"  {obj_type}: {stats[obj_type]['total_actions']} actions\n")

            f.write(f"\nPoint cloud files saved in: {output_dir}\n")
            f.write("File format: {scene_id}_{object_name}_{part_name}_{start_frame}_{end_frame}.npy\n")
            f.write("Data shape: (n_frames, 500, 3)\n")

        print(f"Summary saved to: {summary_file}")
    else:
        print("\nStatistics generation completed. Remove --stats_only flag to extract point clouds.")


if __name__ == "__main__":
    main()