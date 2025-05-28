import pickle
import numpy as np
import pandas as pd
import argparse
import os
import sys
from pprint import pprint


def pretty_print(obj, max_depth=3, max_items=50, current_depth=0, indent=0):
    """
    递归地美观打印嵌套对象，控制打印深度和每级显示的项目数

    Args:
        obj: 要打印的对象
        max_depth: 最大递归深度
        max_items: 每个容器最多显示的项目数
        current_depth: 当前递归深度
        indent: 当前缩进级别
    """
    prefix = "  " * indent

    # 达到最大深度时简化输出
    if current_depth >= max_depth:
        if isinstance(obj, (list, tuple, np.ndarray)):
            print(f"{prefix}[{type(obj).__name__}] 长度: {len(obj)}, 类型: {type(obj).__name__}")
        elif isinstance(obj, dict):
            print(f"{prefix}[{type(obj).__name__}] 包含 {len(obj)} 项")
        else:
            print(f"{prefix}{type(obj).__name__}: {obj}")
        return

    # 处理不同的数据类型
    if isinstance(obj, dict):
        print(f"{prefix}字典包含 {len(obj)} 项:")
        items = list(obj.items())
        for i, (key, value) in enumerate(items[:max_items]):
            print(f"{prefix}  键 '{key}':")
            pretty_print(value, max_depth, max_items, current_depth + 1, indent + 2)
        if len(obj) > max_items:
            print(f"{prefix}  ... 还有 {len(obj) - max_items} 项未显示")

    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__}包含 {len(obj)} 项:")
        for i, item in enumerate(obj[:max_items]):
            print(f"{prefix}  索引 {i}:")
            pretty_print(item, max_depth, max_items, current_depth + 1, indent + 2)
        if len(obj) > max_items:
            print(f"{prefix}  ... 还有 {len(obj) - max_items} 项未显示")

    elif isinstance(obj, np.ndarray):
        print(f"{prefix}NumPy数组: 形状{obj.shape}, 类型{obj.dtype}")
        if obj.size <= max_items:
            if obj.ndim <= 2 and obj.size <= 100:  # 对于小型1D或2D数组显示值
                print(f"{prefix}  值:\n{obj}")
            else:
                print(f"{prefix}  第一个元素: {obj.flat[0]}")
                print(f"{prefix}  最后一个元素: {obj.flat[-1]}")
        else:
            print(f"{prefix}  前 {min(5, obj.size)} 个元素: {obj.flat[:5]}")
            print(f"{prefix}  ... (大型数组)")

    elif isinstance(obj, pd.DataFrame):
        print(f"{prefix}DataFrame: {obj.shape[0]}行 x {obj.shape[1]}列")
        print(f"{prefix}  列: {list(obj.columns)}")
        if len(obj) <= 10:
            print(f"{prefix}  值:\n{obj}")
        else:
            print(f"{prefix}  前 5 行:\n{obj.head()}")

    else:
        print(f"{prefix}{type(obj).__name__}: {obj}")


def read_pickle_file(file_path, max_depth=3, max_items=50):
    """
    读取并打印pickle文件内容

    Args:
        file_path: pickle文件路径
        max_depth: 最大递归深度
        max_items: 每级最多显示的项目数
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在。")
            return False

        # 检查文件是否是pickle文件
        if not file_path.endswith('.pkl'):
            print(f"警告: 文件 '{file_path}' 可能不是pickle文件。")

        # 读取pickle文件
        print(f"读取文件: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 显示基本信息
        print(f"\n文件: {os.path.basename(file_path)}")
        print(f"数据类型: {type(data)}")

        # 详细打印内容
        print("\n文件内容:")
        print("=" * 60)
        pretty_print(data, max_depth=max_depth, max_items=max_items)

        return True

    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return False


def list_pickle_files(directory):
    """列出目录中的所有pickle文件"""
    pickle_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pickle_files.append(os.path.join(root, file))

    if pickle_files:
        print(f"在 {directory} 中找到 {len(pickle_files)} 个pickle文件:")
        for i, file in enumerate(pickle_files):
            print(f"{i + 1}. {file}")
    else:
        print(f"在 {directory} 中没有找到pickle文件。")

    return pickle_files


def interactive_mode(directory=None):
    """交互式模式，允许用户选择目录中的pickle文件"""
    if directory is None:
        directory = input("请输入要搜索的目录 (默认为当前目录): ") or "."

    pickle_files = list_pickle_files(directory)

    if not pickle_files:
        return

    while True:
        choice = input("\n请输入要查看的文件编号 (1-{}) 或 'q' 退出: ".format(len(pickle_files)))

        if choice.lower() == 'q':
            break

        try:
            file_idx = int(choice) - 1
            if 0 <= file_idx < len(pickle_files):
                max_depth = int(input("显示的最大递归深度 (默认为3): ") or "3")
                max_items = int(input("每级显示的最大项目数 (默认为50): ") or "50")
                read_pickle_file(pickle_files[file_idx], max_depth, max_items)
            else:
                print("无效的选择，请输入1到{}之间的数字。".format(len(pickle_files)))
        except ValueError:
            print("请输入有效的数字。")


def main():
    parser = argparse.ArgumentParser(description="读取并打印Pickle文件的内容")
    parser.add_argument("file_path", nargs="?", help="要读取的pickle文件路径")
    parser.add_argument("--max-depth", type=int, default=3, help="最大递归深度 (默认: 3)")
    parser.add_argument("--max-items", type=int, default=50, help="每级显示的最大项目数 (默认: 50)")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式模式，列出目录中所有pickle文件")
    parser.add_argument("--directory", "-d", help="交互式模式的搜索目录")

    args = parser.parse_args()

    if args.interactive or args.directory:
        interactive_mode(args.directory)
    elif args.file_path:
        read_pickle_file(args.file_path, args.max_depth, args.max_items)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
