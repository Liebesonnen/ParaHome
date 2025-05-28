import os
import sys
import numpy as np
import open3d as o3d
import argparse
import glob
import matplotlib.pyplot as plt
from pathlib import Path


def list_motion_files(data_dir):
    """
    列出目录中所有的运动序列文件
    """
    # 查找所有点云序列文件
    point_cloud_files = glob.glob(os.path.join(data_dir, "*_sequence_points.ply"))
    mesh_files = glob.glob(os.path.join(data_dir, "*_sequence.ply"))
    motion_files = glob.glob(os.path.join(data_dir, "*_motion.npy"))
    joint_files = glob.glob(os.path.join(data_dir, "*_joints.npy"))

    print("找到以下文件：")
    print("\n点云序列文件:")
    for i, f in enumerate(point_cloud_files):
        print(f"{i + 1}. {os.path.basename(f)}")

    print("\n网格序列文件:")
    for i, f in enumerate(mesh_files):
        print(f"{i + 1}. {os.path.basename(f)}")

    print("\n运动数据文件:")
    for i, f in enumerate(motion_files):
        print(f"{i + 1}. {os.path.basename(f)}")

    print("\n关节数据文件:")
    for i, f in enumerate(joint_files):
        print(f"{i + 1}. {os.path.basename(f)}")

    return point_cloud_files, mesh_files, motion_files, joint_files


def visualize_point_cloud(point_cloud_file):
    """
    可视化点云序列
    """
    print(f"可视化点云: {os.path.basename(point_cloud_file)}")

    # 读取点云文件
    pcd = o3d.io.read_point_cloud(point_cloud_file)

    # 创建坐标系来显示方向
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 可视化
    o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                      window_name=f"动作序列: {os.path.basename(point_cloud_file)}",
                                      width=1024, height=768)


def visualize_mesh(mesh_file):
    """
    可视化网格序列
    """
    print(f"可视化网格: {os.path.basename(mesh_file)}")

    # 读取网格文件
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    # 创建坐标系来显示方向
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 可视化
    o3d.visualization.draw_geometries([mesh, coordinate_frame],
                                      window_name=f"动作序列: {os.path.basename(mesh_file)}",
                                      width=1024, height=768)


def visualize_joint_data(joint_file):
    """
    可视化关节数据
    """
    print(f"可视化关节数据: {os.path.basename(joint_file)}")

    # 读取关节数据
    joint_data = np.load(joint_file)

    plt.figure(figsize=(10, 6))

    # 如果是多维数据，为每个维度绘制一条线
    if joint_data.ndim > 1 and joint_data.shape[1] > 1:
        for i in range(joint_data.shape[1]):
            plt.plot(joint_data[:, i], label=f'关节 {i + 1}')
        plt.legend()
    else:
        plt.plot(joint_data)

    action_name = os.path.basename(joint_file).split('_joints.npy')[0]
    obj_name = action_name.split('_')[-1] if '_' in action_name else 'object'

    plt.title(f'动作 "{action_name}" 中 {obj_name} 的关节角度变化')
    plt.xlabel('帧')
    plt.ylabel('关节值 (弧度/米)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def animate_motion(motion_file, scan_root):
    """
    使用运动数据文件创建动画
    """
    print(f"动画展示: {os.path.basename(motion_file)}")

    # 提取物体信息
    filename = os.path.basename(motion_file)
    parts = filename.replace('_motion.npy', '').split('_')

    # 动作名和物体部分
    action_name = '_'.join(parts[:-2])
    obj_name = parts[-2]
    part_name = parts[-1]

    # 加载变换矩阵
    transforms = np.load(motion_file)

    # 查找和加载对应的网格
    mesh_path = os.path.join(scan_root, obj_name, "simplified", part_name + ".obj")
    if not os.path.exists(mesh_path):
        print(f"找不到网格文件: {mesh_path}")
        return

    # 加载网格
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"动画: {action_name} - {obj_name}_{part_name}", width=1024, height=768)

    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coordinate_frame)

    # 添加网格
    vis.add_geometry(mesh)

    # 设置初始视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    # 动画循环
    try:
        print("按 'q' 退出动画")
        frame_count = len(transforms)
        frame_idx = 0

        while True:
            # 应用变换
            transform = transforms[frame_idx]
            mesh.transform(np.linalg.inv(transforms[max(0, frame_idx - 1)]))  # 先撤销上一帧的变换
            mesh.transform(transform)  # 应用当前帧的变换

            # 更新可视化
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            # 进到下一帧
            frame_idx = (frame_idx + 1) % frame_count

            # 添加一点延迟
            import time
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="可视化铰接物体运动数据")
    parser.add_argument("--data_dir", default="motion_data", help="运动数据所在目录")
    parser.add_argument("--scan_root", default="data/scan", help="原始扫描数据所在目录")
    parser.add_argument("--mode", choices=["list", "point_cloud", "mesh", "joint", "animate"],
                        default="list", help="运行模式：列出文件、可视化点云、可视化网格、可视化关节数据或动画")
    parser.add_argument("--file", help="要可视化的特定文件")
    args = parser.parse_args()

    # 确保路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    if not os.path.isabs(args.data_dir):
        data_dir = os.path.join(repo_root, args.data_dir)
    else:
        data_dir = args.data_dir

    if not os.path.isabs(args.scan_root):
        scan_root = os.path.join(repo_root, args.scan_root)
    else:
        scan_root = args.scan_root

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误：找不到数据目录 {data_dir}")
        return

    # 根据模式执行相应操作
    if args.mode == "list":
        point_cloud_files, mesh_files, motion_files, joint_files = list_motion_files(data_dir)

    elif args.mode == "point_cloud":
        if args.file:
            file_path = args.file if os.path.isabs(args.file) else os.path.join(data_dir, args.file)
            if os.path.exists(file_path) and file_path.endswith("_sequence_points.ply"):
                visualize_point_cloud(file_path)
            else:
                print(f"错误：文件 {file_path} 不存在或不是点云文件")
        else:
            point_cloud_files, _, _, _ = list_motion_files(data_dir)
            if point_cloud_files:
                visualize_point_cloud(point_cloud_files[0])
            else:
                print("没有找到点云文件")

    elif args.mode == "mesh":
        if args.file:
            file_path = args.file if os.path.isabs(args.file) else os.path.join(data_dir, args.file)
            if os.path.exists(file_path) and file_path.endswith("_sequence.ply"):
                visualize_mesh(file_path)
            else:
                print(f"错误：文件 {file_path} 不存在或不是网格文件")
        else:
            _, mesh_files, _, _ = list_motion_files(data_dir)
            if mesh_files:
                visualize_mesh(mesh_files[0])
            else:
                print("没有找到网格文件")

    elif args.mode == "joint":
        if args.file:
            file_path = args.file if os.path.isabs(args.file) else os.path.join(data_dir, args.file)
            if os.path.exists(file_path) and file_path.endswith("_joints.npy"):
                visualize_joint_data(file_path)
            else:
                print(f"错误：文件 {file_path} 不存在或不是关节数据文件")
        else:
            _, _, _, joint_files = list_motion_files(data_dir)
            if joint_files:
                visualize_joint_data(joint_files[0])
            else:
                print("没有找到关节数据文件")

    elif args.mode == "animate":
        if args.file:
            file_path = args.file if os.path.isabs(args.file) else os.path.join(data_dir, args.file)
            if os.path.exists(file_path) and file_path.endswith("_motion.npy"):
                animate_motion(file_path, scan_root)
            else:
                print(f"错误：文件 {file_path} 不存在或不是运动数据文件")
        else:
            _, _, motion_files, _ = list_motion_files(data_dir)
            if motion_files:
                animate_motion(motion_files[0], scan_root)
            else:
                print("没有找到运动数据文件")


if __name__ == "__main__":
    main()
