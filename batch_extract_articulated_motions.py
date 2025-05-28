# import os
# import sys
# import json
# import pickle
# import argparse
# import numpy as np
# from pathlib import Path
# import open3d as o3d
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# from collections import defaultdict
#
# # Define root repository path
# ROOT_REPOSITORY = os.path.dirname(os.path.abspath(__file__))
# SCAN_ROOT = os.path.join(ROOT_REPOSITORY, "data/scan")
#
# # 目标物体类型
# TARGET_OBJECTS = ['washing machine', 'chair', 'drawer', 'microwave', 'refrigerator', 'trashbin']
#
#
# def get_object_keywords():
#     """Define keywords for each target object type"""
#     return {
#         'washing machine': ['washing', 'washer', 'machine'],
#         'chair': ['chair', 'seat'],
#         'drawer': ['drawer'],
#         'microwave': ['microwave', 'oven'],
#         'refrigerator': ['refrigerator', 'fridge'],
#         'trashbin': ['trash', 'bin', 'garbage']
#     }
#
#
# def identify_object_type_from_name(obj_name, object_keywords):
#     """Identify which target object type an object belongs to"""
#     obj_name_lower = obj_name.lower()
#     for obj_type, keywords in object_keywords.items():
#         if any(keyword in obj_name_lower for keyword in keywords):
#             return obj_type
#     return None
#
#
# def identify_object_type_from_action(action_name, object_keywords):
#     """Identify which target object type is involved in an action"""
#     action_lower = action_name.lower()
#     for obj_type, keywords in object_keywords.items():
#         if any(keyword in action_lower for keyword in keywords):
#             return obj_type
#     return None
#
#
# def analyze_all_scenes_for_objects(scenes_root, target_objects, object_keywords):
#     """
#     分析所有场景中与指定物体相关的动作
#     """
#     # 统计信息
#     stats = initialize_stats()
#
#     # 遍历所有场景
#     processed_scenes = 0
#     for scene_id in range(1, 208):  # s1 to s207
#         scene_name = f"s{scene_id}"
#         scene_path = os.path.join(scenes_root, scene_name)
#
#         if not os.path.exists(scene_path):
#             continue
#
#         processed_scenes += 1
#         if processed_scenes % 20 == 0:
#             print(f"Analyzing scene {scene_name}... ({processed_scenes}/207)")
#
#         # 解析动作标注
#         actions = parse_text_annotations(scene_path)
#         if not actions:
#             continue
#
#         # 加载场景中的物体信息
#         object_in_scene_path = os.path.join(scene_path, "object_in_scene.json")
#         if not os.path.exists(object_in_scene_path):
#             continue
#
#         with open(object_in_scene_path, "r") as f:
#             objects_in_scene = json.load(f)
#
#         # 分析每个动作
#         for action_name, start_frame, end_frame in actions:
#             # 识别动作中涉及的目标物体类型
#             target_object_type = identify_object_type_from_action(action_name, object_keywords)
#
#             # 如果动作名称中没有找到目标物体，跳过
#             if not target_object_type:
#                 continue
#
#             # 找到场景中属于该类型的所有物体
#             articulated_objects = []
#             for scene_obj in objects_in_scene:
#                 obj_type = identify_object_type_from_name(scene_obj, object_keywords)
#                 if obj_type == target_object_type:
#                     articulated_objects.append(scene_obj)
#
#             if not articulated_objects:
#                 continue
#
#             # 更新统计信息
#             stats[target_object_type]['total_actions'] += 1
#             stats[target_object_type]['scenes'].add(scene_name)
#             stats[target_object_type]['actions'].append({
#                 'scene_id': scene_name,
#                 'action_name': action_name,
#                 'start_frame': start_frame,
#                 'end_frame': end_frame,
#                 'objects': articulated_objects,
#                 'scene_path': scene_path
#             })
#
#     print(f"Total processed scenes: {processed_scenes}")
#     return stats
#
#
# def parse_text_annotations(scene_root):
#     """Parse the text annotations file to get action frame ranges."""
#     # Try both possible filenames
#     possible_filenames = ["text_annotations.json", "text_annotation.json"]
#     annotations_path = None
#
#     for filename in possible_filenames:
#         path = os.path.join(scene_root, filename)
#         if os.path.exists(path):
#             annotations_path = path
#             break
#
#     if annotations_path is None:
#         return []
#
#     try:
#         with open(annotations_path, "r") as f:
#             annotations = json.load(f)
#
#         actions = []
#         for frame_range, action_name in annotations.items():
#             try:
#                 start_frame, end_frame = map(int, frame_range.split())
#                 actions.append((action_name, start_frame, end_frame))
#             except ValueError:
#                 continue
#
#         return actions
#     except Exception as e:
#         return []
#
#
# def initialize_stats():
#     """Initialize statistics dictionary"""
#     stats = {}
#     for obj_type in TARGET_OBJECTS:
#         stats[obj_type] = {
#             'total_actions': 0,
#             'scenes': set(),  # Will be converted to list later
#             'actions': []
#         }
#     return stats
#
#
# def extract_articulated_point_clouds_batch(stats, output_dir):
#     """
#     批量提取所有相关动作的点云数据
#     """
#     total_extractions = sum(len(stats[obj]['actions']) for obj in TARGET_OBJECTS)
#     current_extraction = 0
#
#     for obj_type in TARGET_OBJECTS:
#         if not stats[obj_type]['actions']:
#             print(f"No actions found for {obj_type}")
#             continue
#
#         print(f"\n=== Extracting point clouds for {obj_type} ({len(stats[obj_type]['actions'])} actions) ===")
#
#         for action_info in stats[obj_type]['actions']:
#             current_extraction += 1
#             scene_path = action_info["scene_path"]
#             start_frame = action_info["start_frame"]
#             end_frame = action_info["end_frame"]
#             scene_id = action_info["scene_id"]
#             objects = action_info["objects"]
#             action_name = action_info["action_name"]
#
#             print(
#                 f"[{current_extraction}/{total_extractions}] Processing {scene_id}: {action_name[:50]}... (frames {start_frame}-{end_frame})")
#
#             try:
#                 extract_single_action_point_clouds(
#                     scene_path, start_frame, end_frame, objects, scene_id, action_name, output_dir
#                 )
#             except Exception as e:
#                 print(f"Error processing {scene_id} {obj_type}: {e}")
#                 continue
#
#
# def extract_single_action_point_clouds(scene_path, start_frame, end_frame, objects, scene_id, action_name, output_dir):
#     """
#     提取单个动作的点云数据
#     """
#     # 加载对象变换数据
#     object_transform_path = os.path.join(scene_path, "object_transformations.pkl")
#     if not os.path.exists(object_transform_path):
#         print(f"Object transformations not found: {object_transform_path}")
#         return
#
#     with open(object_transform_path, "rb") as f:
#         object_transforms = pickle.load(f)
#
#     # 获取有效帧
#     valid_frames = [f for f in range(start_frame, end_frame + 1) if f in object_transforms]
#     if not valid_frames:
#         print(f"No valid frames found for {scene_id} between {start_frame} and {end_frame}")
#         return
#
#     # 为每个物体的每个部分提取点云
#     for obj_name in objects:
#         # 查找对象的所有部分
#         object_parts = []
#         for frame in valid_frames:
#             for key in object_transforms[frame].keys():
#                 if key.startswith(obj_name + "_"):
#                     if key not in object_parts:
#                         object_parts.append(key)
#
#         if not object_parts:
#             print(f"Object '{obj_name}' not found in {scene_id} for frames {start_frame}-{end_frame}")
#             continue
#
#         # 为每个部分提取点云
#         for part_full_name in object_parts:
#             part_name = part_full_name.split("_")[1]
#
#             # 加载网格
#             mesh_path = os.path.join(SCAN_ROOT, obj_name, "simplified", part_name + ".obj")
#             if not os.path.exists(mesh_path):
#                 print(f"Mesh not found: {mesh_path}")
#                 continue
#
#             # 加载网格并采样点
#             try:
#                 mesh = o3d.io.read_triangle_mesh(mesh_path)
#                 if len(mesh.vertices) == 0:
#                     print(f"Empty mesh: {mesh_path}")
#                     continue
#
#                 points = np.asarray(mesh.sample_points_uniformly(number_of_points=500).points)
#             except Exception as e:
#                 print(f"Error loading mesh {mesh_path}: {e}")
#                 continue
#
#             # 收集变换矩阵
#             part_frames = []
#             part_transforms = []
#
#             for frame in valid_frames:
#                 if part_full_name in object_transforms[frame]:
#                     part_frames.append(frame)
#                     part_transforms.append(object_transforms[frame][part_full_name])
#
#             if not part_frames:
#                 continue
#
#             # 应用变换
#             transformed_points = []
#             for transform in part_transforms:
#                 try:
#                     # 确保变换矩阵的格式正确
#                     transform = np.array(transform)
#                     if transform.shape != (4, 4):
#                         print(f"Invalid transformation matrix shape: {transform.shape}")
#                         continue
#
#                     rotation = transform[:3, :3]
#                     translation = transform[:3, 3]
#                     transformed = np.dot(points, rotation.T) + translation
#                     transformed_points.append(transformed)
#                 except Exception as e:
#                     print(f"Error applying transformation: {e}")
#                     continue
#
#             if not transformed_points:
#                 continue
#
#             # 保存点云数据
#             point_cloud = np.array(transformed_points)  # shape: (n_frames, 500, 3)
#
#             frame_range_str = f"{start_frame}_{end_frame}"
#             output_file = os.path.join(output_dir, f"{scene_id}_{obj_name}_{part_name}_{frame_range_str}.npy")
#
#             np.save(output_file, point_cloud)
#             print(f"Saved: {os.path.basename(output_file)}, shape: {point_cloud.shape}")
#
#
# def save_stats(stats, output_dir):
#     """Save statistics to JSON file"""
#     # Convert sets to lists for JSON serialization
#     stats_for_json = {}
#     for obj_type, data in stats.items():
#         stats_for_json[obj_type] = {
#             'total_actions': data['total_actions'],
#             'unique_scenes': list(data['scenes']),
#             'scene_count': len(data['scenes']),
#             'actions': data['actions']
#         }
#
#     # Add summary
#     summary = {
#         'total_target_actions': sum(data['total_actions'] for data in stats.values()),
#         'total_unique_scenes_with_target_actions': len(set().union(*[data['scenes'] for data in stats.values()])),
#         'object_types_processed': TARGET_OBJECTS
#     }
#
#     result = {
#         'summary': summary,
#         'by_object_type': stats_for_json
#     }
#
#     stats_file = os.path.join(output_dir, "target_object_actions_statistics.json")
#     with open(stats_file, 'w', encoding='utf-8') as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)
#
#     print(f"\nStatistics saved to: {stats_file}")
#     return result
#
#
# def print_summary(stats):
#     """Print summary statistics"""
#     print("\n" + "=" * 60)
#     print("SUMMARY STATISTICS")
#     print("=" * 60)
#
#     total_actions = 0
#     all_scenes = set()
#
#     for obj_type in TARGET_OBJECTS:
#         data = stats[obj_type]
#         total_actions += data['total_actions']
#         all_scenes.update(data['scenes'])
#
#         print(f"\n{obj_type.upper()}:")
#         print(f"  Total actions: {data['total_actions']}")
#         print(f"  Scenes involved: {len(data['scenes'])}")
#         if data['scenes']:
#             scenes_list = sorted(data['scenes'], key=lambda x: int(x[1:]))  # Sort by scene number
#             print(f"  Scene IDs: {', '.join(scenes_list[:10])}{' ...' if len(scenes_list) > 10 else ''}")
#
#     print(f"\nOVERALL:")
#     print(f"  Total target object actions: {total_actions}")
#     print(f"  Total scenes with target actions: {len(all_scenes)}")
#     print(f"  Scene coverage: {len(all_scenes)}/207 ({len(all_scenes) / 207 * 100:.1f}%)")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Batch extract articulated object point clouds from all scenes")
#     parser.add_argument("--scenes_root", default="data/seq", help="Root directory containing all scenes")
#     parser.add_argument("--output_dir", default="output_batch", help="Directory to save output files")
#     parser.add_argument("--stats_only", action="store_true",
#                         help="Only generate statistics, don't extract point clouds")
#
#     args = parser.parse_args()
#
#     # 目标物体列表
#     target_objects = TARGET_OBJECTS
#     object_keywords = get_object_keywords()
#
#     # 确保路径正确
#     scenes_root = args.scenes_root
#     if not os.path.isabs(scenes_root):
#         scenes_root = os.path.join(ROOT_REPOSITORY, scenes_root)
#
#     output_dir = args.output_dir
#     if not os.path.isabs(output_dir):
#         output_dir = os.path.join(ROOT_REPOSITORY, output_dir)
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     print("=== Analyzing all scenes for target objects ===")
#     print(f"Target objects: {target_objects}")
#     print(f"Scenes root: {scenes_root}")
#     print(f"Output directory: {output_dir}")
#
#     # 分析所有场景
#     print("Scanning all scenes...")
#     stats = analyze_all_scenes_for_objects(scenes_root, target_objects, object_keywords)
#
#     # 保存统计信息
#     save_stats(stats, output_dir)
#     print_summary(stats)
#
#     if not args.stats_only:
#         print(f"\n=== Extracting point clouds ===")
#         extract_articulated_point_clouds_batch(stats, output_dir)
#         print(f"\nPoint cloud extraction completed!")
#
#         # 生成总结报告
#         summary_file = os.path.join(output_dir, "extraction_summary.txt")
#         with open(summary_file, "w") as f:
#             f.write("ParaHome Articulated Objects Extraction Summary\n")
#             f.write("=" * 50 + "\n\n")
#             f.write(f"Total scenes processed: s1 to s207\n")
#             f.write(f"Target objects: {', '.join(target_objects)}\n")
#             f.write(f"Total actions found: {sum(stats[obj]['total_actions'] for obj in target_objects)}\n\n")
#
#             f.write("Actions per object:\n")
#             for obj_type in target_objects:
#                 f.write(f"  {obj_type}: {stats[obj_type]['total_actions']} actions\n")
#
#             f.write(f"\nPoint cloud files saved in: {output_dir}\n")
#             f.write("File format: {scene_id}_{object_name}_{part_name}_{start_frame}_{end_frame}.npy\n")
#             f.write("Data shape: (n_frames, 500, 3)\n")
#
#         print(f"Summary saved to: {summary_file}")
#     else:
#         print("\nStatistics generation completed. Remove --stats_only flag to extract point clouds.")
#
#
# if __name__ == "__main__":
#     main()

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

# 目标动作类型
TARGET_ACTIONS = [
    'open microwave',
    'close microwave',
    'open refrigerator',
    'close refrigerator',
    'open washing machine',
    'close washing machine',
    'sit on chair',
    'get up from chair',
    'throw away in trash',
    'take out from drawer'
]


def get_action_keywords():
    """Define keywords for each target action type"""
    return {
        'open microwave': {
            'action_words': ['open'],
            'object_words': ['microwave']
        },
        'close microwave': {
            'action_words': ['close'],
            'object_words': ['microwave']
        },
        'open refrigerator': {
            'action_words': ['open'],
            'object_words': ['refrigerator', 'fridge']
        },
        'close refrigerator': {
            'action_words': ['close'],
            'object_words': ['refrigerator', 'fridge']
        },
        'open washing machine': {
            'action_words': ['open'],
            'object_words': ['washing', 'washer', 'machine']
        },
        'close washing machine': {
            'action_words': ['close'],
            'object_words': ['washing', 'washer', 'machine']
        },
        'sit on chair': {
            'action_words': ['sit', 'seat'],
            'object_words': ['chair', 'seat']
        },
        'get up from chair': {
            'action_words': ['get up', 'stand up', 'stand', 'rise', 'get off'],
            'object_words': ['chair', 'seat']
        },
        'throw away in trash': {
            'action_words': ['throw', 'dispose', 'put', 'drop'],
            'object_words': ['trash', 'garbage', 'bin', 'waste']
        },
        'take out from drawer': {
            'action_words': ['take out', 'pull out', 'get', 'retrieve', 'take'],
            'object_words': ['drawer']
        }
    }


def get_object_keywords():
    """Define object keywords for matching scene objects"""
    return {
        'microwave': ['microwave'],
        'refrigerator': ['refrigerator', 'fridge'],
        'washing machine': ['washing', 'washer', 'machine'],
        'chair': ['chair', 'seat', 'deskchair'],  # Include deskchair
        'trashbin': ['trash', 'bin', 'garbage', 'waste'],
        'drawer': ['drawer']
    }


def identify_action_type_from_text(action_text, action_keywords):
    """Identify which target action type matches the action text"""
    action_lower = action_text.lower()

    for action_type, keywords in action_keywords.items():
        action_words = keywords['action_words']
        object_words = keywords['object_words']

        # Check if any action word is present
        has_action = False
        for action_word in action_words:
            if ' ' in action_word:  # Multi-word phrases like "get up"
                if action_word in action_lower:
                    has_action = True
                    break
            else:  # Single words
                if action_word in action_lower:
                    has_action = True
                    break

        # Check if any object word is present
        has_object = any(obj_word in action_lower for obj_word in object_words)

        # Special handling for chair actions - be more flexible
        if 'chair' in action_type:
            if action_type == 'sit on chair':
                # Look for sitting-related words
                sit_indicators = ['sit', 'seat', 'sitting']
                chair_indicators = ['chair', 'seat']
                has_sit = any(word in action_lower for word in sit_indicators)
                has_chair = any(word in action_lower for word in chair_indicators)
                if has_sit and has_chair:
                    return action_type
            elif action_type == 'get up from chair':
                # Look for standing/getting up words
                standup_indicators = ['stand', 'get up', 'rise', 'getting up', 'stood up', 'get off']
                chair_indicators = ['chair', 'seat']
                has_standup = any(word in action_lower for word in standup_indicators)
                has_chair = any(word in action_lower for word in chair_indicators)
                if has_standup and has_chair:
                    return action_type

        # Special handling for trash actions
        elif 'trash' in action_type:
            trash_indicators = ['throw', 'dispose', 'put', 'drop', 'toss', 'place']
            container_indicators = ['trash', 'garbage', 'bin', 'waste', 'rubbish']
            has_throw = any(word in action_lower for word in trash_indicators)
            has_container = any(word in action_lower for word in container_indicators)
            if has_throw and has_container:
                return action_type

        # For other actions, use the standard logic
        elif has_action and has_object:
            return action_type

    return None


def identify_object_type_from_name(obj_name, object_keywords):
    """Identify which object type an object belongs to"""
    obj_name_lower = obj_name.lower()
    for obj_type, keywords in object_keywords.items():
        if any(keyword in obj_name_lower for keyword in keywords):
            return obj_type
    return None


def get_moving_parts(scene_path, start_frame, end_frame, obj_name):
    """
    Identify which parts of an object are moving by analyzing joint states or transformation changes
    """
    # Special handling for chairs - always return base since chair movement is holistic
    if 'chair' in obj_name.lower():
        return ['base']

    # Try to load joint states first
    joint_states_path = os.path.join(scene_path, "joint_states.pkl")
    if os.path.exists(joint_states_path):
        try:
            with open(joint_states_path, "rb") as f:
                joint_states = pickle.load(f)

            # Check if object has joint state changes
            moving_parts = set()
            for frame in range(start_frame, end_frame + 1):
                if frame in joint_states and obj_name in joint_states[frame]:
                    # If joint states exist and change, the articulated part is moving
                    joint_data = joint_states[frame][obj_name]
                    if hasattr(joint_data, '__len__') and len(joint_data) > 0:
                        # Assume part1 is the moving part for articulated objects
                        moving_parts.add('part1')

            if moving_parts:
                return list(moving_parts)
        except Exception as e:
            print(f"Error reading joint states: {e}")

    # Fallback: analyze transformation changes
    object_transform_path = os.path.join(scene_path, "object_transformations.pkl")
    if not os.path.exists(object_transform_path):
        return []

    with open(object_transform_path, "rb") as f:
        object_transforms = pickle.load(f)

    # Find all parts of the object
    all_parts = set()
    for frame in range(start_frame, end_frame + 1):
        if frame in object_transforms:
            for key in object_transforms[frame].keys():
                if key.startswith(obj_name + "_"):
                    part_name = key.split("_")[1]
                    all_parts.add(part_name)

    # Analyze transformation changes for each part
    moving_parts = []
    for part in all_parts:
        part_key = f"{obj_name}_{part}"
        transforms = []

        for frame in range(start_frame, end_frame + 1):
            if frame in object_transforms and part_key in object_transforms[frame]:
                transforms.append(object_transforms[frame][part_key])

        if len(transforms) > 1:
            # Calculate transformation differences
            movement_detected = False
            for i in range(1, len(transforms)):
                diff = np.array(transforms[i]) - np.array(transforms[i - 1])
                # Check if there's significant movement (translation or rotation)
                if np.linalg.norm(diff[:3, 3]) > 0.001 or np.linalg.norm(diff[:3, :3]) > 0.001:
                    movement_detected = True
                    break

            if movement_detected:
                moving_parts.append(part)

    # Default fallback rules based on object type
    if not moving_parts:
        if 'microwave' in obj_name:
            moving_parts = ['part1']  # door
        elif 'refrigerator' in obj_name:
            moving_parts = ['part1']  # door
        elif 'washing' in obj_name:
            moving_parts = ['part1']  # door
        elif 'drawer' in obj_name:
            moving_parts = ['part1', 'part2']  # drawer parts
        elif 'trashbin' in obj_name:
            moving_parts = ['part1']  # lid
        elif 'chair' in obj_name:
            moving_parts = ['base']  # for chair, we keep base

    return moving_parts


def analyze_all_scenes_for_actions(scenes_root, target_actions, action_keywords, object_keywords):
    """
    分析所有场景中与指定动作相关的行为
    """
    # 统计信息
    stats = initialize_stats()

    # Debug: 记录所有找到的动作
    all_found_actions = []

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
            # 识别动作类型
            action_type = identify_action_type_from_text(action_name, action_keywords)

            if action_type:
                all_found_actions.append({
                    'scene': scene_name,
                    'action_type': action_type,
                    'action_name': action_name,
                    'objects_in_scene': objects_in_scene
                })

            if not action_type:
                continue

            # 根据动作类型找到相关物体
            relevant_objects = []
            if 'microwave' in action_type:
                target_obj_type = 'microwave'
            elif 'refrigerator' in action_type:
                target_obj_type = 'refrigerator'
            elif 'washing machine' in action_type:
                target_obj_type = 'washing machine'
            elif 'chair' in action_type:
                target_obj_type = 'chair'
            elif 'trash' in action_type:
                target_obj_type = 'trashbin'
            elif 'drawer' in action_type:
                target_obj_type = 'drawer'
            else:
                continue

            # 在场景中找到该类型的物体
            for scene_obj in objects_in_scene:
                obj_type = identify_object_type_from_name(scene_obj, object_keywords)
                if obj_type == target_obj_type:
                    relevant_objects.append(scene_obj)

            if not relevant_objects:
                # Debug: 记录没有找到相关物体的情况
                print(f"  No {target_obj_type} found in {scene_name} for action: {action_name}")
                continue

            # 更新统计信息
            stats[action_type]['total_actions'] += 1
            stats[action_type]['scenes'].add(scene_name)
            stats[action_type]['actions'].append({
                'scene_id': scene_name,
                'action_name': action_name,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'objects': relevant_objects,
                'scene_path': scene_path
            })

    print(f"Total processed scenes: {processed_scenes}")

    # Debug: 输出一些找到的动作示例
    print(f"\nDEBUG: Found {len(all_found_actions)} potential target actions")
    chair_actions = [a for a in all_found_actions if 'chair' in a['action_type']]
    trash_actions = [a for a in all_found_actions if 'trash' in a['action_type']]

    print(f"Chair-related actions found: {len(chair_actions)}")
    if chair_actions:
        for action in chair_actions[:5]:  # Show first 5
            print(f"  {action['scene']}: {action['action_name']} -> {action['action_type']}")

    print(f"Trash-related actions found: {len(trash_actions)}")
    if trash_actions:
        for action in trash_actions[:5]:  # Show first 5
            print(f"  {action['scene']}: {action['action_name']} -> {action['action_type']}")

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
    for action_type in TARGET_ACTIONS:
        stats[action_type] = {
            'total_actions': 0,
            'scenes': set(),  # Will be converted to list later
            'actions': []
        }
    return stats


def extract_articulated_point_clouds_batch(stats, output_dir):
    """
    批量提取所有相关动作的点云数据
    """
    total_extractions = sum(len(stats[action]['actions']) for action in TARGET_ACTIONS)
    current_extraction = 0

    for action_type in TARGET_ACTIONS:
        if not stats[action_type]['actions']:
            print(f"No actions found for {action_type}")
            continue

        print(f"\n=== Extracting point clouds for {action_type} ({len(stats[action_type]['actions'])} actions) ===")

        for action_info in stats[action_type]['actions']:
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
                print(f"Error processing {scene_id} {action_type}: {e}")
                continue


def extract_single_action_point_clouds(scene_path, start_frame, end_frame, objects, scene_id, action_name, output_dir):
    """
    提取单个动作的点云数据，只保存运动的部分
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

    # 为每个物体提取点云
    for obj_name in objects:
        # 识别运动的部分
        moving_parts = get_moving_parts(scene_path, start_frame, end_frame, obj_name)

        if not moving_parts:
            print(f"No moving parts identified for {obj_name} in {scene_id}")
            continue

        print(f"  Moving parts for {obj_name}: {moving_parts}")

        # 为每个运动部分提取点云
        for part_name in moving_parts:
            part_full_name = f"{obj_name}_{part_name}"

            # 检查该部分是否存在于变换数据中
            part_exists = False
            for frame in valid_frames:
                if part_full_name in object_transforms[frame]:
                    part_exists = True
                    break

            if not part_exists:
                print(f"Part {part_full_name} not found in transformations")
                continue

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
    for action_type, data in stats.items():
        stats_for_json[action_type] = {
            'total_actions': data['total_actions'],
            'unique_scenes': list(data['scenes']),
            'scene_count': len(data['scenes']),
            'actions': data['actions']
        }

    # Add summary
    summary = {
        'total_target_actions': sum(data['total_actions'] for data in stats.values()),
        'total_unique_scenes_with_target_actions': len(set().union(*[data['scenes'] for data in stats.values()])),
        'action_types_processed': TARGET_ACTIONS
    }

    result = {
        'summary': summary,
        'by_action_type': stats_for_json
    }

    stats_file = os.path.join(output_dir, "target_action_statistics.json")
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

    for action_type in TARGET_ACTIONS:
        data = stats[action_type]
        total_actions += data['total_actions']
        all_scenes.update(data['scenes'])

        print(f"\n{action_type.upper()}:")
        print(f"  Total actions: {data['total_actions']}")
        print(f"  Scenes involved: {len(data['scenes'])}")
        if data['scenes']:
            scenes_list = sorted(data['scenes'], key=lambda x: int(x[1:]))  # Sort by scene number
            print(f"  Scene IDs: {', '.join(scenes_list[:10])}{' ...' if len(scenes_list) > 10 else ''}")

    print(f"\nOVERALL:")
    print(f"  Total target actions: {total_actions}")
    print(f"  Total scenes with target actions: {len(all_scenes)}")
    print(f"  Scene coverage: {len(all_scenes)}/207 ({len(all_scenes) / 207 * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Batch extract specific articulated object actions from all scenes")
    parser.add_argument("--scenes_root", default="data/seq", help="Root directory containing all scenes")
    parser.add_argument("--output_dir", default="output_specific_actions", help="Directory to save output files")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only generate statistics, don't extract point clouds")

    args = parser.parse_args()

    # 目标动作列表
    target_actions = TARGET_ACTIONS
    action_keywords = get_action_keywords()
    object_keywords = get_object_keywords()

    # 确保路径正确
    scenes_root = args.scenes_root
    if not os.path.isabs(scenes_root):
        scenes_root = os.path.join(ROOT_REPOSITORY, scenes_root)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(ROOT_REPOSITORY, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print("=== Analyzing all scenes for target actions ===")
    print(f"Target actions: {target_actions}")
    print(f"Scenes root: {scenes_root}")
    print(f"Output directory: {output_dir}")

    # 分析所有场景
    print("Scanning all scenes...")
    stats = analyze_all_scenes_for_actions(scenes_root, target_actions, action_keywords, object_keywords)

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
            f.write("ParaHome Specific Action Extraction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total scenes processed: s1 to s207\n")
            f.write(f"Target actions: {', '.join(target_actions)}\n")
            f.write(f"Total actions found: {sum(stats[action]['total_actions'] for action in target_actions)}\n\n")

            f.write("Actions per type:\n")
            for action_type in target_actions:
                f.write(f"  {action_type}: {stats[action_type]['total_actions']} actions\n")

            f.write(f"\nPoint cloud files saved in: {output_dir}\n")
            f.write("File format: {scene_id}_{object_name}_{part_name}_{start_frame}_{end_frame}.npy\n")
            f.write("Data shape: (n_frames, 500, 3)\n")
            f.write("Note: Only moving parts are saved (excluding base parts except for chairs)\n")

        print(f"Summary saved to: {summary_file}")
    else:
        print("\nStatistics generation completed. Remove --stats_only flag to extract point clouds.")


if __name__ == "__main__":
    main()