from __init__ import *  # 从__init__.py导入所有内容

import os  # 导入操作系统模块，用于文件路径操作
import json  # 导入JSON模块，用于处理JSON数据
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import argparse  # 导入命令行参数解析模块
import torch  # 导入PyTorch深度学习框架
from pathlib import Path  # 导入Path类，用于更现代的文件路径处理
import open3d as o3d  # 导入Open3D库，用于3D数据处理和可视化
import numpy as np  # 导入NumPy库，用于数值计算

from utils import makeTpose, rotation_6d_to_matrix, BodyMaker, HandMaker, \
    simpleViewer, get_stickman, get_stickhand  # 从utils模块导入各种工具函数

SCAN_ROOT = os.path.join(ROOT_REPOSITORY, "data/scan")  # 设置扫描数据的根目录路径

import torch  # 再次导入PyTorch（这是冗余的）

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置设备为GPU（如果可用）或CPU

if __name__ == "__main__":  # 主程序入口
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("--scene_root", default="data/seq/s01", help="target path, e.g : ")  # 添加场景根目录参数
    parser.add_argument("--start_frame", default=0, type=int, help="Render start frame number")  # 添加起始帧参数
    parser.add_argument("--end_frame", default=100000, type=int, help="Render end frame number")  # 添加结束帧参数
    parser.add_argument("--run", default=False, action='store_true',
                        help='If set, viewer will show start_frame scene at specific view')  # 添加运行模式参数
    parser.add_argument("--fromort", default=False, action='store_true',
                        help='Make skeleton from orientation, transform to ')  # 添加从方向创建骨架参数
    parser.add_argument("--ego", action='store_true')  # 添加第一人称视角参数
    parser.add_argument("--smplx", action='store_true')  # 添加使用SMPLX人体模型参数
    parser.add_argument("--show_joints", action='store_true', help='Show articulated object joints')
    args = parser.parse_args()  # 解析命令行参数

    root = os.path.join(ROOT_REPOSITORY, args.scene_root)  # 构建完整的场景根目录路径
    camera_dir = root + "/cam_param"  # 设置相机参数目录

    head_tip_position = pickle.load(open(root + "/head_tips.pkl", "rb"))  # 加载头部顶点位置数据
    if args.fromort:  # 如果使用方向数据创建骨架
        bone_vector = pickle.load(open(root + "/bone_vectors.pkl", "rb"))  # 加载骨骼向量数据
        bodyTpose, HandTpose = makeTpose(bone_vector)  # 创建身体和手部的T-pose

        body_ort = pickle.load(open(root + "/body_joint_orientations.pkl", "rb"))  # 加载身体关节方向数据
        hand_ort = pickle.load(open(root + "/hand_joint_orientations.pkl", "rb"))  # 加载手部关节方向数据
        body_rotmat = rotation_6d_to_matrix(torch.tensor(body_ort, dtype=torch.float32)).numpy()  # 将6D旋转表示转换为旋转矩阵
        hand_rotmat = rotation_6d_to_matrix(torch.tensor(hand_ort, dtype=torch.float32)).numpy()  # 将手部6D旋转表示转换为旋转矩阵

        bodymaker = BodyMaker(bodyTpose)  # 创建身体生成器
        handmaker = HandMaker(HandTpose)  # 创建手部生成器
        body_joint = bodymaker(torch.tensor(body_ort, dtype=torch.float32)).numpy()  # 使用方向数据生成身体关节位置
        hand_joint = handmaker(torch.tensor(hand_ort, dtype=torch.float32)).numpy()  # 使用方向数据生成手部关节位置

        lhand = np.einsum('FMN, FJN -> FJM', body_rotmat[:, 14], hand_joint[:, 0]) + body_joint[:, 14][:, None,
                                                                                     :]  # 计算左手关节位置
        rhand = np.einsum('FMN, FJN -> FJM', body_rotmat[:, 10], hand_joint[:, 1]) + body_joint[:, 10][:, None,
                                                                                     :]  # 计算右手关节位置
        joint_root_fixed = np.concatenate([body_joint, lhand, rhand], axis=1)  # 合并身体和手部关节数据

        body_global_trans = pickle.load(open(root + "/body_global_transform.pkl", "rb"))  # 加载身体全局变换数据
        joint_rgb = np.einsum('FMN, FJN->FJM', body_global_trans[:, :3, :3], joint_root_fixed) + body_global_trans[:,
                                                                                                 :3, 3][:, None,
                                                                                                 :]  # 应用全局变换到关节位置
    else:  # 如果不从方向数据创建骨架
        joint_rgb = pickle.load(open(root + "/joint_positions.pkl", "rb"))  # 直接加载关节位置数据

        if args.ego:  # 如果使用第一人称视角
            body_global_trans = pickle.load(open(root + "/body_global_transform.pkl", "rb"))  # 加载身体全局变换数据
            body_ort = pickle.load(open(root + "/body_joint_orientations.pkl", "rb"))  # 加载身体关节方向数据
            body_rotmat = rotation_6d_to_matrix(torch.tensor(body_ort, dtype=torch.float32)).numpy()  # 将6D旋转转换为旋转矩阵

    if args.smplx:  # 如果使用SMPLX人体模型
        import smplx  # 导入SMPLX模块

        smplroot = root.replace('seq', 'smplx_seq')  # 设置SMPLX数据路径
        SMPLX_MODEL_PATH = '/home/jisoo/data2/git_repo/smplx/transfer_data/models'  # 设置SMPLX模型路径

        smplx_params = pickle.load(open(smplroot + "/smplx_params.pkl", "rb"))  # 加载SMPLX参数
        smplx_pose = pickle.load(open(smplroot + "/smplx_pose.pkl", "rb"))  # 加载SMPLX姿势数据

        smplx_beta, gender = smplx_params['beta'].to(device), smplx_params['gender']  # 提取beta参数和性别信息
        body_pose = smplx_pose['body_pose'].reshape((-1, 21, 3)).to(device)  # 提取并重塑身体姿势数据
        global_orient = smplx_pose['global_orient'].to(device)  # 提取全局方向数据
        transl = smplx_pose['transl'].to(device)  # 提取平移数据
        hand_pose = smplx_pose['hand_pose'].reshape((-1, 30, 3)).to(device)  # 提取并重塑手部姿势数据
        lhand_pose = hand_pose[:, :15, :].to(device)  # 分离左手姿势数据
        rhand_pose = hand_pose[:, 15:, :].to(device)  # 分离右手姿势数据

        body_model = smplx.create(model_path=SMPLX_MODEL_PATH,  # 创建SMPLX模型
                                  model_type="smplx",
                                  flat_hand_mean=True,
                                  use_pca=False,
                                  num_betas=20,
                                  num_expression_coeffs=10,
                                  gender=gender,
                                  ext='pkl').to(device)

        smplx_faces = body_model.faces  # 获取SMPLX模型的面信息

    frame_length = joint_rgb.shape[0]  # 获取帧数量

    os.path.dirname(__file__)  # 获取当前文件的目录（未使用）
    obj_color = json.load(open(os.path.join(ROOT_REPOSITORY, "visualize/color.json"), "r"))  # 加载对象颜色配置
    object_transform = pickle.load(open(root + "/object_transformations.pkl", "rb"))  # 加载对象变换数据

    print("------Reading Objects------")  # 打印正在读取对象的信息
    initialized, mesh_dict = dict(), dict()  # 初始化字典，用于存储初始化状态和网格数据
    obj_in_scene = json.load(open(Path(root, "object_in_scene.json"), "r"))  # 加载场景中的对象列表
    # 加载对象
    for objn in obj_in_scene:  # 遍历场景中的每个对象
        for pn in ["base", "part1", "part2"]:  # 遍历对象的各个部分
            meshpath = Path(SCAN_ROOT, objn, "simplified", pn + ".obj")  # 构建网格文件路径
            if meshpath.exists():  # 如果文件存在
                keyn = objn + "_" + pn  # 创建键名
                initialized[keyn] = False  # 设置初始化状态为False
                m = o3d.io.read_triangle_mesh(str(meshpath))  # 读取三角网格
                m.paint_uniform_color(obj_color[objn][pn])  # 设置网格颜色
                m.compute_vertex_normals()  # 计算顶点法线
                mesh_dict[keyn] = m  # 存储网格到字典中

    # 设置相机参数
    width, height = 1600, 800  # 设置视窗宽度和高度
    if args.ego:  # 如果使用第一人称视角
        view = o3d.camera.PinholeCameraParameters()  # 创建针孔相机参数
        camera_matrix = np.eye(3, dtype=np.float64)  # 创建相机矩阵
        f = 520  # 设置焦距
        camera_matrix[0, 0] = f  # 设置x方向焦距
        camera_matrix[1, 1] = f  # 设置y方向焦距
        camera_matrix[0, 2] = width / 2  # 设置主点x坐标
        camera_matrix[1, 2] = height / 2  # 设置主点y坐标
        view.intrinsic.intrinsic_matrix = camera_matrix  # 设置内参矩阵
        view.intrinsic.width, view.intrinsic.height = width, height  # 设置视窗尺寸
    else:  # 否则
        view = None  # 不使用特定视图

    vis = simpleViewer("Render Scene", 1600, 800, [], view)  # 创建简单查看器

    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)  # 创建全局坐标系
    vis.add_geometry({"name": "global", "geometry": global_coord})  # 添加全局坐标系到查看器
    vis.add_plane()  # 添加平面到查看器
    for fn in range(args.start_frame, min(args.end_frame + 1, frame_length)):  # 遍历指定帧范围
        """
        Head tip position 
        """
        if fn not in object_transform:  # 如果当前帧没有对象变换数据
            continue  # 跳过此帧

        cur_human_joints = joint_rgb[fn]  # 获取当前帧的人体关节位置

        cur_object_pose = object_transform[fn]  # 获取当前帧的对象姿势

        if args.smplx:  # 如果使用SMPLX模型
            output_glb = body_model(betas=smplx_beta,  # 使用SMPLX模型生成输出
                                    return_verts=True,  # 返回顶点
                                    body_pose=body_pose[fn:fn + 1],  # 1X21X3  使用当前帧的身体姿势
                                    left_hand_pose=lhand_pose[fn:fn + 1],  # 1X15X3  使用当前帧的左手姿势
                                    right_hand_pose=rhand_pose[fn:fn + 1],  # 1X15X3  使用当前帧的右手姿势
                                    global_orient=global_orient[fn:fn + 1],  # 1X3  使用当前帧的全局方向
                                    transl=transl[fn:fn + 1]  # + tl,  使用当前帧的平移
                                    )
            verts_glb = output_glb.vertices[0]  # 获取顶点
            jts_glb = output_glb.joints[0]  # 获取关节

            bmesh = o3d.geometry.TriangleMesh()  # 创建三角网格
            bmesh_vertices = verts_glb.detach().cpu().numpy()  # 将顶点转换为NumPy数组
            bmesh.vertices = o3d.utility.Vector3dVector(bmesh_vertices)  # 设置网格顶点
            bmesh.triangles = o3d.utility.Vector3iVector(smplx_faces)  # 设置网格面
            bmesh.compute_vertex_normals()  # 计算顶点法线
            bmesh.paint_uniform_color([0.4, 0.4, 0.4])  # 设置网格颜色

            vis.add_geometry({"name": "human", "geometry": bmesh})  # 添加人体网格到查看器
        else:  # 如果不使用SMPLX模型
            bmesh = get_stickman(cur_human_joints[:23], head_tip_position[fn] if not args.ego else None)  # 创建人体棍状模型
            hmesh = get_stickhand(cur_human_joints[23:48]) + get_stickhand(cur_human_joints[48:])  # 创建手部棍状模型
            bmesh.compute_vertex_normals()  # 计算身体顶点法线
            hmesh.compute_vertex_normals()  # 计算手部顶点法线

            vis.add_geometry({"name": "human", "geometry": bmesh + hmesh})  # 添加人体和手部网格到查看器

        # 按类别处理对象
        for inst_name, loaded in initialized.items():  # 遍历初始化状态字典
            if loaded and inst_name in cur_object_pose:  # 如果对象已加载且当前帧中存在
                vis.transform(inst_name, cur_object_pose[inst_name])  # 应用变换到对象
            elif loaded and not inst_name in cur_object_pose:  # 如果对象已加载但当前帧中不存在
                vis.remove_geometry(inst_name)  # 从查看器中移除对象
                initialized[inst_name] = False  # 更新初始化状态
            elif not loaded and inst_name in cur_object_pose:  # 如果对象未加载但当前帧中存在
                vis.add_geometry({"name": inst_name, "geometry": mesh_dict[inst_name]})  # 添加对象到查看器
                vis.transform(inst_name, cur_object_pose[inst_name])  # 应用变换
                initialized[inst_name] = True  # 更新初始化状态
            elif not loaded and not inst_name in cur_object_pose:  # 如果对象未加载且当前帧中不存在
                continue  # 跳过处理

        if fn == args.start_frame:  # 如果是起始帧
            vis.main_vis.reset_camera_to_default()  # 重置相机到默认状态

        if args.ego:  # 如果使用第一人称视角
            head_posinrgb = cur_human_joints[5]  # 获取头部位置
            head_rotinrgb = body_global_trans[fn, :3, :3] @ body_rotmat[fn, 6]  # 计算头部旋转
            # 改变轴方向
            head_rotinrgb = np.stack([-head_rotinrgb[:, 1], -head_rotinrgb[:, 2], head_rotinrgb[:, 0]],
                                     axis=1)  # 调整头部旋转轴
            head_Trgb = np.eye(4, dtype=np.float64)  # 创建变换矩阵
            head_Trgb[:3, :3] = head_rotinrgb  # 设置旋转部分
            head_Trgb[:3, 3] = head_posinrgb  # 设置平移部分
            extrinsic_matrix = np.linalg.inv(head_Trgb)  # 计算外参矩阵
            vis.setupcamera(extrinsic_matrix)  # 设置相机外参

        if args.run:  # 如果是运行模式
            vis.run()  # 运行查看器
        else:  # 否则
            vis.tick()  # 更新帧

        vis.remove_geometry("human")  # 从查看器中移除人体模型