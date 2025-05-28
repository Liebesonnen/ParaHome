import os
import json
import pickle
import numpy as np
from pathlib import Path
import glob

# 原始数据
original_data = {
    "drawer": {
        "part1": {
            "axis": [-0.8588028693571086, -0.00946019297345899, -0.5122188363706494],
            "pivot": [0.10756733590642843]
        },
        "part2": {
            "axis": [-0.8648810775361201, 0.0010988751708692235, -0.5019756111538872],
            "pivot": [0.16000230530987383]
        }
    },
    "sink": {
        "part1": {
            "axis": [0.7908714991634659, -0.15935230219002566, -0.5908714882253763],
            "pivot": [0.039308, -0.480645, -0.371972]
        },
        "part2": {
            "axis": [0.7942666255055025, -0.16021758173913775, -0.5860638652141631],
            "pivot": [0.1298, 0.3878, -0.460375]
        }
    },
    "refrigerator": {
        "part1": {
            "axis": [0.529561429306864, 0.053152541384484144, 0.8466046892941487],
            "pivot": [0.585, 0.228, 0.2958]
        },
        "part2": {
            "axis": [0.529561429306864, 0.053152541384484144, 0.8466046892941487],
            "pivot": [0.585, 0.228, 0.2958]
        }
    },
    "trashbin": {
        "part1": {
            "axis": [0.4772596943829716, -0.29430636953294426, -0.8280138555421707],
            "pivot": [-0.0456, -0.045861, -0.070583]
        }
    },
    "washingmachine": {
        "part1": {
            "axis": [-0.7807595133781433, -0.06500383466482162, 0.6219111084938049],
            "pivot": [-0.17421495914459229, -0.20119653642177582, -0.24751949310302734]
        }
    },
    "gasstove": {
        "part1": {
            "axis": [0.03967991591025114, -0.9842499538313702, 0.17227168268842444],
            "pivot": [0.118917, -0.238991, 0.163482]
        },
        "part2": {
            "axis": [0.04274325218622727, -0.983703434588261, 0.17464411576633077],
            "pivot": [-0.142084, -0.276986, 0.0224294]
        }
    },
    "microwave": {
        "part1": {
            "axis": [-1.2761681079864502, 0.33918681740760803, 1.0404295921325684],
            "pivot": [0.06971579045057297, -0.2857133150100708, -0.022020254284143448]
        }
    },
    "laptop": {
        "part1": {
            "axis": [-2.04838490486145, 1.642666220664978, 3.4366250038146973],
            "pivot": [1.6836799383163452, -1.232885718345642, -2.815009832382202]
        }
    }
}


def transform_axis_and_pivot_for_scene(scene_root, scene_name):
    """
    为单个场景转换axis和pivot数据

    Args:
        scene_root: 场景根目录路径
        scene_name: 场景名称（如 's1', 's2' 等）

    Returns:
        dict: 转换后的数据，如果失败则返回None
    """

    # 加载object_transformations.pkl文件
    object_transform_path = os.path.join(scene_root, "object_transformations.pkl")

    if not os.path.exists(object_transform_path):
        print(f"警告：场景 {scene_name} - 找不到文件 {object_transform_path}")
        return None

    try:
        with open(object_transform_path, "rb") as f:
            object_transforms = pickle.load(f)
    except Exception as e:
        print(f"警告：场景 {scene_name} - 加载文件失败: {e}")
        return None

    # 寻找第0帧或者最早的帧
    frame_keys = sorted(object_transforms.keys())
    if not frame_keys:
        print(f"警告：场景 {scene_name} - 没有找到任何帧数据")
        return None

    # 优先使用第0帧，如果不存在则使用最早的帧
    target_frame = 0 if 0 in frame_keys else frame_keys[0]
    frame_transforms = object_transforms[target_frame]

    # 存储转换后的数据
    transformed_data = {}

    print(f"处理场景 {scene_name} (使用第{target_frame}帧数据)")

    # 遍历每个物体
    for obj_name, obj_data in original_data.items():
        base_key = f"{obj_name}_base"

        if base_key not in frame_transforms:
            print(f"  警告：{scene_name} - 第{target_frame}帧中找不到 {base_key}，跳过该物体")
            continue

        # 获取base的转换矩阵
        base_transform = np.array(frame_transforms[base_key])

        # 提取旋转矩阵和平移向量
        rotation_matrix = base_transform[:3, :3]
        translation_vector = base_transform[:3, 3]

        transformed_data[obj_name] = {}

        # 转换每个部分的axis和pivot
        for part_name, part_data in obj_data.items():
            # 转换axis（方向向量，只需要旋转）
            original_axis = np.array(part_data["axis"])
            transformed_axis = rotation_matrix @ original_axis

            # 转换pivot（点，需要旋转和平移）
            original_pivot = np.array(part_data["pivot"])

            # 如果pivot是1D的（单个值），我们假设它是沿着某个轴的距离
            if len(original_pivot) == 1:
                # 对于1D pivot，我们假设它是沿着axis方向的距离
                # 创建一个3D点：原点 + 距离 * 单位axis向量
                if np.linalg.norm(original_axis) > 0:
                    unit_axis = original_axis / np.linalg.norm(original_axis)
                    pivot_3d = original_pivot[0] * unit_axis
                else:
                    pivot_3d = np.array([original_pivot[0], 0, 0])

                transformed_pivot_3d = rotation_matrix @ pivot_3d + translation_vector

                # 计算转换后的距离（投影到转换后的axis上）
                if np.linalg.norm(transformed_axis) > 0:
                    unit_transformed_axis = transformed_axis / np.linalg.norm(transformed_axis)
                    transformed_pivot_distance = np.dot(transformed_pivot_3d, unit_transformed_axis)
                    transformed_pivot = [transformed_pivot_distance]
                else:
                    transformed_pivot = [np.linalg.norm(transformed_pivot_3d)]
            else:
                # 对于3D pivot
                transformed_pivot = rotation_matrix @ original_pivot + translation_vector

            transformed_data[obj_name][part_name] = {
                "axis": transformed_axis.tolist(),
                "pivot": transformed_pivot.tolist() if hasattr(transformed_pivot, 'tolist') else transformed_pivot
            }

    return transformed_data


def process_all_scenes(data_root):
    """
    处理所有场景 s1-s207

    Args:
        data_root: 数据根目录（包含seq文件夹）

    Returns:
        dict: 所有场景的转换结果
    """
    seq_root = os.path.join(data_root, "seq")

    if not os.path.exists(seq_root):
        print(f"错误：找不到seq目录 {seq_root}")
        return None

    # 查找所有场景目录
    scene_dirs = []
    for i in range(1, 208):  # s1 to s207
        scene_name = f"s{i}"
        scene_path = os.path.join(seq_root, scene_name)
        if os.path.exists(scene_path):
            scene_dirs.append((scene_name, scene_path))

    print(f"找到 {len(scene_dirs)} 个场景目录")

    # 存储所有场景的转换结果
    all_scenes_data = {}

    # 统计信息
    success_count = 0
    failed_count = 0

    # 处理每个场景
    for scene_name, scene_path in scene_dirs:
        transformed_data = transform_axis_and_pivot_for_scene(scene_path, scene_name)

        if transformed_data is not None:
            all_scenes_data[scene_name] = transformed_data
            success_count += 1
            print(f"  ✓ {scene_name} 处理成功")
        else:
            failed_count += 1
            print(f"  ✗ {scene_name} 处理失败")

    print(f"\n处理完成：成功 {success_count} 个，失败 {failed_count} 个")

    return all_scenes_data


def save_results(all_scenes_data, output_file):
    """
    保存所有场景的转换结果

    Args:
        all_scenes_data: 所有场景的转换数据
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_scenes_data, f, indent=2, ensure_ascii=False)

        print(f"\n所有场景的转换结果已保存到: {output_file}")

        # 打印一些统计信息
        total_scenes = len(all_scenes_data)
        total_objects = 0
        total_parts = 0

        for scene_name, scene_data in all_scenes_data.items():
            total_objects += len(scene_data)
            for obj_name, obj_data in scene_data.items():
                total_parts += len(obj_data)

        print(f"统计信息：")
        print(f"  - 总场景数: {total_scenes}")
        print(f"  - 总物体数: {total_objects}")
        print(f"  - 总部件数: {total_parts}")

    except Exception as e:
        print(f"保存文件失败: {e}")


def main():
    # 设置数据根目录（请根据你的实际路径修改）
    data_root = "/common/homes/all/uksqc_chen/projects/control/ParaHome/data"  # 包含seq文件夹的根目录

    print("ParaHome批量数据转换工具")
    print("=" * 60)
    print(f"数据根目录: {data_root}")

    # 检查路径是否存在
    if not os.path.exists(data_root):
        print(f"错误：路径 {data_root} 不存在")
        print("请修改data_root变量为正确的路径")
        return

    # 处理所有场景
    all_scenes_data = process_all_scenes(data_root)

    if all_scenes_data:
        # 保存结果
        output_file = "all_scenes_transformed_axis_pivot_data.json"
        save_results(all_scenes_data, output_file)

        # 显示第一个场景的示例数据
        if all_scenes_data:
            first_scene = list(all_scenes_data.keys())[0]
            print(f"\n示例数据 (场景 {first_scene}):")
            print("-" * 40)
            print(json.dumps(all_scenes_data[first_scene], indent=2, ensure_ascii=False)[:500] + "...")
    else:
        print("所有场景处理失败")


if __name__ == "__main__":
    main()