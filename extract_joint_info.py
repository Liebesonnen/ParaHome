import pickle
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from pprint import pprint
import datetime


def convert_to_serializable(obj):
    """递归地将不可JSON序列化的对象转换为可序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        # 处理自定义对象
        return {key: convert_to_serializable(value) for key, value in obj.__dict__.items()}
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # 无法转换的类型，转为字符串
        return f"<<无法序列化的类型 {type(obj).__name__}::{str(obj)}>>"


def load_joint_info(file_path):
    """加载joint_info.pkl文件"""
    try:
        with open(file_path, 'rb') as f:
            joint_info = pickle.load(f)
        print(f"成功从{file_path}加载数据")
        return joint_info
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None


def analyze_and_save_structure(joint_info, output_dir):
    """分析并保存结构信息"""
    if joint_info is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    structure_info = {}

    # 分析整体结构
    structure_info["类型"] = str(type(joint_info))

    if isinstance(joint_info, dict):
        structure_info["结构"] = "字典"
        structure_info["键数量"] = len(joint_info)
        structure_info["键列表"] = list(joint_info.keys())
    elif isinstance(joint_info, list):
        structure_info["结构"] = "列表"
        structure_info["项目数量"] = len(joint_info)
        if len(joint_info) > 0:
            structure_info["第一项类型"] = str(type(joint_info[0]))
    else:
        structure_info["结构"] = "其他类型"

    # 保存结构信息
    with open(os.path.join(output_dir, "结构信息.json"), 'w', encoding='utf-8') as f:
        json.dump(structure_info, f, indent=2, ensure_ascii=False)

    print(f"结构信息已保存到 {os.path.join(output_dir, 'structure.json')}")


def save_all_data(joint_info, output_dir):
    """保存所有数据，尽可能完整地保留结构"""
    if joint_info is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 保存为JSON (转换不可序列化的数据类型)
    json_data = convert_to_serializable(joint_info)
    json_path = os.path.join(output_dir, "all_data.json")

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"所有数据已保存到 {json_path}")
    except Exception as e:
        print(f"保存JSON时出错: {e}")

    # 保存为pickled格式以完全保留原始结构
    pkl_path = os.path.join(output_dir, "all_data_original.pkl")
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(joint_info, f)
        print(f"原始数据格式已保存到 {pkl_path}")
    except Exception as e:
        print(f"保存Pickle时出错: {e}")

    # 生成人类可读文本报告
    txt_path = os.path.join(output_dir, "data_report.txt")
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("JOINT_INFO 文件数据报告\n")
            f.write("========================\n\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if isinstance(joint_info, dict):
                f.write(f"数据类型: 字典, 包含 {len(joint_info)} 个条目\n\n")
                f.write("键列表:\n")
                for key in joint_info.keys():
                    f.write(f"- {key}\n")

                f.write("\n样本数据:\n")
                sample_keys = list(joint_info.keys())[:5]  # 取前5个键作为样本
                for key in sample_keys:
                    f.write(f"\n--- 键: {key} ---\n")
                    value = joint_info[key]
                    if isinstance(value, dict):
                        for k, v in list(value.items())[:10]:  # 限制输出条目
                            f.write(f"  {k}: {str(v)[:100]}{'...' if len(str(v)) > 100 else ''}\n")
                    elif isinstance(value, list):
                        f.write(f"  列表, 长度: {len(value)}\n")
                        if len(value) > 0:
                            f.write(f"  第一个元素: {str(value[0])[:100]}{'...' if len(str(value[0])) > 100 else ''}\n")
                    else:
                        f.write(f"  值: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}\n")

            elif isinstance(joint_info, list):
                f.write(f"数据类型: 列表, 包含 {len(joint_info)} 项\n\n")
                if len(joint_info) > 0:
                    f.write("样本数据:\n")
                    for i, item in enumerate(joint_info[:5]):  # 取前5项作为样本
                        f.write(f"\n--- 项目 {i} ---\n")
                        if isinstance(item, dict):
                            for k, v in list(item.items())[:10]:
                                f.write(f"  {k}: {str(v)[:100]}{'...' if len(str(v)) > 100 else ''}\n")
                        else:
                            f.write(f"  值: {str(item)[:200]}{'...' if len(str(item)) > 200 else ''}\n")

            else:
                f.write(f"数据类型: {type(joint_info)}\n")
                f.write(f"值: {str(joint_info)[:500]}{'...' if len(str(joint_info)) > 500 else ''}\n")

        print(f"人类可读报告已保存到 {txt_path}")
    except Exception as e:
        print(f"生成文本报告时出错: {e}")


def extract_articulated_objects(joint_info, output_dir):
    """尝试提取关节物体信息并保存"""
    if joint_info is None or not isinstance(joint_info, dict):
        return

    # 尝试识别关节物体 (基于常见的关节物体名称)
    articulated_keywords = [
        "cabinet", "refrigerator", "microwave", "pot", "drawer",
        "washing", "gasstove", "laptop", "kettle", "door", "joint"
    ]

    articulated_objects = {}

    # 根据关键词搜索
    for key in joint_info.keys():
        if any(keyword in key.lower() for keyword in articulated_keywords):
            articulated_objects[key] = joint_info[key]

    # 如果没有找到，尝试在值中搜索类型信息
    if not articulated_objects:
        for key, value in joint_info.items():
            if isinstance(value, dict) and any(
                    kw in str(value).lower() for kw in ["joint", "revolute", "prismatic", "articulated"]
            ):
                articulated_objects[key] = value

    # 保存识别的关节物体
    if articulated_objects:
        json_path = os.path.join(output_dir, "铰接物体.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(convert_to_serializable(articulated_objects), f, indent=2, ensure_ascii=False)
            print(f"识别出 {len(articulated_objects)} 个铰接物体，已保存到 {json_path}")
        except Exception as e:
            print(f"保存铰接物体时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="提取并保存joint_info.pkl中的所有信息")
    parser.add_argument("--input", default="data/joint_info.pkl", help="joint_info.pkl文件路径")
    parser.add_argument("--output", default="joint_info_完整数据", help="输出目录")
    args = parser.parse_args()

    # 处理相对路径
    if not os.path.isabs(args.input):
        # 尝试在当前目录和上级目录查找
        possible_paths = [
            args.input,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), args.input),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.input)
        ]

        for path in possible_paths:
            if os.path.exists(path):
                input_path = path
                break
        else:
            input_path = args.input  # 默认使用输入路径
    else:
        input_path = args.input

    # 加载数据
    joint_info = load_joint_info(input_path)

    # 分析并保存所有数据
    analyze_and_save_structure(joint_info, args.output)
    save_all_data(joint_info, args.output)
    extract_articulated_objects(joint_info, args.output)

    print("\n处理完成！所有信息已保存到目录:", args.output)


if __name__ == "__main__":
    main()