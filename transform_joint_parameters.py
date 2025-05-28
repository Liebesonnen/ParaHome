import os
import json
import pickle
import numpy as np
from pathlib import Path

# 你提供的原始数据
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


def transform_axis_and_pivot(scene_root):
    """
    根据object_transformations.pkl中第0帧的base转换矩阵对axis和pivot进行转换

    Args:
        scene_root: 场景根目录路径
    """

    # 加载object_transformations.pkl文件
    object_transform_path = os.path.join(scene_root, "object_transformations.pkl")

    if not os.path.exists(object_transform_path):
        print(f"错误：找不到文件 {object_transform_path}")
        return

    with open(object_transform_path, "rb") as f:
        object_transforms = pickle.load(f)

    # 获取第0帧的数据
    if 10 not in object_transforms:
        print("错误：第0帧数据不存在")
        return

    frame_0_transforms = object_transforms[10]

    # 存储转换后的数据
    transformed_data = {}

    print("=" * 60)
    print("根据第0帧base转换矩阵转换axis和pivot")
    print("=" * 60)

    # 遍历每个物体
    for obj_name, obj_data in original_data.items():
        base_key = f"{obj_name}_base"

        if base_key not in frame_0_transforms:
            print(f"警告：第0帧中找不到 {base_key} 的转换矩阵，跳过该物体")
            continue

        # 获取base的转换矩阵
        base_transform = np.array(frame_0_transforms[base_key])

        print(f"\n物体: {obj_name}")
        print(f"Base转换矩阵:")
        print(base_transform)

        # 提取旋转矩阵和平移向量
        rotation_matrix = base_transform[:3, :3]
        translation_vector = base_transform[:3, 3]

        transformed_data[obj_name] = {}

        # 转换每个部分的axis和pivot
        for part_name, part_data in obj_data.items():
            print(f"\n  部分: {part_name}")

            # 转换axis（方向向量，只需要旋转）
            original_axis = np.array(part_data["axis"])
            transformed_axis = rotation_matrix @ original_axis

            print(f"    原始axis: {original_axis}")
            print(f"    转换后axis: {transformed_axis}")

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

            print(f"    原始pivot: {original_pivot}")
            print(f"    转换后pivot: {transformed_pivot}")

            transformed_data[obj_name][part_name] = {
                "axis": transformed_axis.tolist(),
                "pivot": transformed_pivot.tolist() if hasattr(transformed_pivot, 'tolist') else transformed_pivot
            }

    return transformed_data


def print_transformed_data(transformed_data):
    """
    打印转换后的数据
    """
    print("\n" + "=" * 60)
    print("转换后的完整数据:")
    print("=" * 60)

    print(json.dumps(transformed_data, indent=2, ensure_ascii=False))


def main():
    # 设置场景路径（请根据你的实际路径修改）
    scene_root = "data/seq/s6"  # 请修改为你的实际路径

    print("ParaHome数据axis和pivot转换工具")
    print(f"场景路径: {scene_root}")

    # 检查路径是否存在
    if not os.path.exists(scene_root):
        print(f"错误：路径 {scene_root} 不存在")
        print("请修改scene_root变量为正确的路径")
        return

    # 执行转换
    transformed_data = transform_axis_and_pivot(scene_root)

    if transformed_data:
        # 打印结果
        print_transformed_data(transformed_data)

        # 保存到文件
        output_file = "transformed_axis_pivot_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)

        print(f"\n转换后的数据已保存到: {output_file}")
    else:
        print("转换失败")


if __name__ == "__main__":
    main()
