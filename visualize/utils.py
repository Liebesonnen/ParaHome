import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
from pytorch3d.transforms import rotation_6d_to_matrix  # 从PyTorch3D导入6D旋转到矩阵的转换函数

import open3d as o3d  # 导入Open3D库，用于3D数据处理和可视化
import numpy as np  # 导入NumPy库，用于数值计算

body_order = {'pHipOrigin': 0,  # 定义身体关节顺序字典，从髋关节原点开始
              'jL5S1': 1,  # 腰椎L5-骶骨S1连接处
              'jL4L3': 2,  # 腰椎L4-L3连接处
              'jL1T12': 3,  # 腰椎L1-胸椎T12连接处
              'jT9T8': 4,  # 胸椎T9-T8连接处
              'jT1C7': 5,  # 胸椎T1-颈椎C7连接处
              'jC1Head': 6,  # 颈椎C1-头部连接处
              'jRightT4Shoulder': 7,  # 右侧T4肩膀连接处
              'jRightShoulder': 8,  # 右肩
              'jRightElbow': 9,  # 右肘
              'jRightWrist': 10,  # 右手腕
              'jLeftT4Shoulder': 11,  # 左侧T4肩膀连接处
              'jLeftShoulder': 12,  # 左肩
              'jLeftElbow': 13,  # 左肘
              'jLeftWrist': 14,  # 左手腕
              'jRightHip': 15,  # 右髋
              'jRightKnee': 16,  # 右膝
              'jRightAnkle': 17,  # 右踝
              'jRightBallFoot': 18,  # 右脚掌
              'jLeftHip': 19,  # 左髋
              'jLeftKnee': 20,  # 左膝
              'jLeftAnkle': 21,  # 左踝
              'jLeftBallFoot': 22}  # 左脚掌

ljt_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4,
             'jLeftFifthCMC': 5, 'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'pLeftFifthTip': 9,
             'jLeftFourthMCP': 10, 'jLeftFourthPIP': 11, 'jLeftFourthDIP': 12, 'pLeftFourthTip': 13,
             'jLeftThirdMCP': 14, 'jLeftThirdPIP': 15, 'jLeftThirdDIP': 16, 'pLeftThirdTip': 17, 'jLeftSecondMCP': 18,
             'jLeftSecondPIP': 19, 'jLeftSecondDIP': 20, 'pLeftSecondTip': 21, 'jLeftFirstMCP': 22, 'jLeftIP': 23,
             'pLeftFirstTip': 24}  # 定义左手关节顺序字典
rjt_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4,
             'jRightFifthCMC': 5, 'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'pRightFifthTip': 9,
             'jRightFourthMCP': 10, 'jRightFourthPIP': 11, 'jRightFourthDIP': 12, 'pRightFourthTip': 13,
             'jRightThirdMCP': 14, 'jRightThirdPIP': 15, 'jRightThirdDIP': 16, 'pRightThirdTip': 17,
             'jRightSecondMCP': 18, 'jRightSecondPIP': 19, 'jRightSecondDIP': 20, 'pRightSecondTip': 21,
             'jRightFirstMCP': 22, 'jRightIP': 23, 'pRightFirstTip': 24}  # 定义右手关节顺序字典
lp_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4,
            'jLeftFifthCMC': 5, 'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'jLeftFourthMCP': 9,
            'jLeftFourthPIP': 10, 'jLeftFourthDIP': 11, 'jLeftThirdMCP': 12, 'jLeftThirdPIP': 13, 'jLeftThirdDIP': 14,
            'jLeftSecondMCP': 15, 'jLeftSecondPIP': 16, 'jLeftSecondDIP': 17, 'jLeftFirstMCP': 18,
            'jLeftIP': 19}  # 定义左手掌关节顺序字典
rp_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4,
            'jRightFifthCMC': 5, 'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'jRightFourthMCP': 9,
            'jRightFourthPIP': 10, 'jRightFourthDIP': 11, 'jRightThirdMCP': 12, 'jRightThirdPIP': 13,
            'jRightThirdDIP': 14, 'jRightSecondMCP': 15, 'jRightSecondPIP': 16, 'jRightSecondDIP': 17,
            'jRightFirstMCP': 18, 'jRightIP': 19}  # 定义右手掌关节顺序字典


def makeTpose(bone_vector):  # 定义创建T姿势的函数
    bodyTpose, handTpose = torch.zeros(22, 3), torch.zeros(2, 24, 3)  # 初始化身体和手部的T姿势张量
    for p_nm, child_lst in bone_vector["body"].items():  # 遍历身体骨骼向量
        for c_nm, bone in child_lst.items():  # 遍历每个父节点的子节点
            if c_nm[-3:] != "Toe":  # 如果不是脚趾节点
                jtidx = body_order[c_nm] - 1  # 获取关节索引
                bodyTpose[jtidx] = torch.tensor(bone, requires_grad=False)  # 设置身体T姿势的关节位置

    for p_nm, child_lst in bone_vector["lhand"].items():  # 遍历左手骨骼向量
        for c_nm, bone in child_lst.items():  # 遍历每个父节点的子节点
            if c_nm[0] != "p" or c_nm[-3:] == "Tip":  # 如果不是点或是指尖
                jtidx = ljt_order[c_nm] - 1  # 获取左手关节索引
                handTpose[0][jtidx] = torch.tensor(bone, requires_grad=False)  # 设置左手T姿势的关节位置

    for p_nm, child_lst in bone_vector["rhand"].items():  # 遍历右手骨骼向量
        for c_nm, bone in child_lst.items():  # 遍历每个父节点的子节点
            if c_nm[0] != "p" or c_nm[-3:] == "Tip":  # 如果不是点或是指尖
                jtidx = (rjt_order[c_nm] - 1)  # 获取右手关节索引
                handTpose[1][jtidx] = torch.tensor(bone, requires_grad=False)  # 设置右手T姿势的关节位置
    return bodyTpose.reshape(-1), handTpose.reshape(2, 72)  # 返回重塑后的身体和手部T姿势


def makeHandMapper():  # 定义创建手部映射器的函数
    base_array = torch.zeros((20, 72, 72), dtype=torch.float32)  # concat  # 初始化基础数组
    base_array[0, :15, :15] = torch.eye(15)  # 设置第一个映射矩阵的左上角为单位矩阵
    # fi+1 : finger number
    for fi in range(5):  # 遍历5个手指
        fi_idx = 4 - fi  # 计算手指索引（从小指到拇指）
        if fi == 0:  # 如果是小指
            # Rwrist
            row_start, row_end = 15 + 12 * fi_idx, 72  # 计算行的起始和结束索引
            col_start = 15 + 12 * fi_idx  # 63  # 计算列的起始索引
            # wrist rot
            base_array[0, 15 + 12 * fi_idx:, :3] = torch.cat([torch.eye(3) for _ in range(3)], dim=0)  # 设置手腕旋转
            # CMC 전용 (专用于掌指关节)
            base_array[fi + 1, row_start:row_end, col_start:col_start + 3] = torch.cat([torch.eye(3) for _ in range(3)],
                                                                                       dim=0)  # 4*fi_idx+1  # 设置CMC关节映射
            base_array[3 * (fi_idx + 2), row_start + 3:row_end, col_start + 3:col_start + 6] = torch.cat(
                [torch.eye(3) for _ in range(2)], dim=0)  # 设置MCP关节映射
            base_array[3 * (fi_idx + 2) + 1, row_start + 6:row_end, col_start + 6:col_start + 9] = torch.eye(
                3)  # 设置PIP关节映射
        else:  # 对于其他手指
            row_start, row_end = 15 + 12 * fi_idx, 15 + 12 * (fi_idx + 1)  # 计算行的起始和结束索引
            col_start = 15 + 12 * fi_idx  # 计算列的起始索引
            # wrist rot
            base_array[0, row_start:row_end, 3 * fi:3 * (fi + 1)] = torch.cat([torch.eye(3) for _ in range(4)],
                                                                              dim=0)  # 设置手腕旋转对该手指的影响
            # Rot CMC 전용 (专用于掌指关节旋转)
            base_array[fi + 1, row_start:row_end, col_start:col_start + 3] = torch.cat([torch.eye(3) for _ in range(4)],
                                                                                       dim=0)  # fi == 4  # 设置CMC关节映射
            base_array[3 * (fi_idx + 2), row_start + 3:row_end, col_start + 3:col_start + 6] = torch.cat(
                [torch.eye(3) for _ in range(3)], dim=0)  # 设置MCP关节映射
            base_array[3 * (fi_idx + 2) + 1, row_start + 6:row_end, col_start + 6:col_start + 9] = torch.cat(
                [torch.eye(3) for _ in range(2)], dim=0)  # 设置PIP关节映射
            base_array[3 * (fi_idx + 2) + 2, row_start + 9:row_end, col_start + 9:col_start + 12] = torch.eye(
                3)  # 设置DIP关节映射
    base_array = torch.cat([base_array[None, :], base_array[None, :]], dim=0)  # 为左右手复制映射矩阵
    return base_array  # 2,20,72,72  # 返回手部映射器


class HandMaker(nn.Module):  # 定义手部生成器类
    def __init__(self, bone_length):  # 初始化方法
        super().__init__()  # 调用父类初始化方法
        """
        bone_length : (48,6 => 2, 72)
        """
        mapped_reltrans = torch.einsum('SJMK, SK -> SJM', makeHandMapper(),
                                       bone_length)  # mapped bonelength  # 计算映射后的相对变换
        mapped_reltrans = mapped_reltrans.reshape(2, 20, 24, 3)  # 重塑为适合处理的形状
        self.mapper = nn.Parameter(mapped_reltrans, requires_grad=False)  # 创建不需要梯度的参数

    def forward(self, orientation):  # 前向传播方法
        """
        Orientation : torch.tensor (F, 40,6) rotation value
        """
        num_frame = orientation.shape[0]  # 获取帧数
        wrist_pos = torch.zeros((num_frame, 2, 1, 3)).to(orientation.device)  # 初始化手腕位置
        mat = rotation_6d_to_matrix(orientation)  # (40,3,3) z xc  # 将6D旋转表示转换为旋转矩阵
        mat = mat.view(num_frame, 2, 20, 3, 3)  # 重塑旋转矩阵

        rel_mapped = torch.einsum('FSRMK, SRJK -> FSRJM', mat, self.mapper)  # # 应用旋转到映射的关节
        acq_joints = torch.sum(rel_mapped, dim=2)  # (2,24,3)  # 累加各层的贡献得到关节位置
        out = torch.cat([wrist_pos, acq_joints], dim=2)  # 拼接手腕位置和其他关节位置
        return out  # 返回完整的手部姿势


def makeBodyMapper():  # 定义创建身体映射器的函数
    base_array = torch.zeros((23, 23 * 3, 22 * 3), dtype=torch.float32)  # concat  # 初始化基础数组
    for i in range(23):  # 遍历所有身体关节
        if i == 0:  # 如果是髋关节原点
            base_array[i][1 * 3:15 * 3, :3] = torch.cat([torch.eye(3) for _ in range(14)], axis=0)  # 设置对脊柱和上肢的影响
            base_array[i][15 * 3:19 * 3, 14 * 3:15 * 3] = torch.cat([torch.eye(3) for _ in range(4)],
                                                                    axis=0)  # 设置对右腿的影响
            base_array[i][19 * 3:23 * 3, 18 * 3:19 * 3] = torch.cat([torch.eye(3) for _ in range(4)],
                                                                    axis=0)  # 设置对左腿的影响
        elif (i >= 1 and i <= 3):  # 如果是腰椎区域
            start_row = i + 1  # 计算行的起始索引
            end_row = 15  # 行的结束索引
            base_array[i][start_row * 3:end_row * 3, 3 * i:3 * (i + 1)] = torch.cat(
                [torch.eye(3) for _ in range(14 - i)], axis=0)  # 设置对上方关节的影响
        elif i == 4:  # jT9T8  # 如果是胸椎T9-T8
            start_row = i + 1  # 计算行的起始索引
            end_row = 7  # 行的结束索引
            base_array[i][start_row * 3:end_row * 3, 3 * i:3 * (i + 1)] = torch.cat([torch.eye(3) for _ in range(2)],
                                                                                    axis=0)  # 设置对颈部的影响
            # shoulders
            base_array[i][end_row * 3:(end_row + 4) * 3, 3 * (i + 2):3 * (i + 3)] = torch.cat(
                [torch.eye(3) for _ in range(4)], axis=0)  # RT4shoulder  # 设置对右肩的影响
            base_array[i][(end_row + 4) * 3:(end_row + 8) * 3, 3 * (i + 6):3 * (i + 7)] = torch.cat(
                [torch.eye(3) for _ in range(4)], axis=0)  # LT4shoulder  # 设置对左肩的影响
        elif i == 5:  # 如果是胸椎T1-颈椎C7
            start_row = i + 1  # 计算行的起始索引
            end_row = 7  # 行的结束索引
            base_array[i][start_row * 3:end_row * 3, 3 * i:3 * (i + 1)] = torch.cat([torch.eye(3) for _ in range(1)],
                                                                                    axis=0)  # 设置对头部的影响
        elif i >= 7:  # 如果是肩部及以下关节
            start_col, start_row = i * 3, (i + 1) * 3  # 计算列和行的起始索引
            num_ident = 3 - (i - 7) % 4  # 计算单位矩阵的数量
            end_row = start_row + 3 * num_ident  # 计算行的结束索引
            if num_ident >= 1:  # 如果需要至少一个单位矩阵
                base_array[i][start_row:end_row, start_col:start_col + 3] = torch.cat(
                    [torch.eye(3) for _ in range(num_ident)], axis=0)  # LT4shoulder  # 设置对子关节的影响
    return base_array  # 返回身体映射器


class BodyMaker(nn.Module):  # 定义身体生成器类
    def __init__(self, bone_trans):  # 初始化方法
        super().__init__()  # 调用父类初始化方法
        """
        bonetrans : (22, 3) => (66,)
        """
        mapped_reltrans = torch.einsum('JMN, N -> JM', makeBodyMapper(), bone_trans)  # 计算映射后的相对变换
        mapped_reltrans = mapped_reltrans.reshape(23, 23, 3)  # 重塑为适合处理的形状
        self.mapper = nn.Parameter(mapped_reltrans, requires_grad=False)  # 创建不需要梯度的参数

    def forward(self, orientation):  # 前向传播方法
        """
        Orientation : torch.tensor (F,23,6) rotation value
        """
        mat = rotation_6d_to_matrix(orientation)  # 将6D旋转表示转换为旋转矩阵
        batched = torch.einsum('FJMN, JLN -> FJLM', mat, self.mapper)  # 应用旋转到映射的关节

        return torch.sum(batched, dim=1)  # , batched # (23,3)  # 返回完整的身体姿势


def getRotation(vec1, vec2):  # 定义获取旋转矩阵的函数
    vec1 = vec1 / np.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2)  # 归一化第一个向量
    vec2 = vec2 / np.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)  # 归一化第二个向量

    n = np.cross(vec1, vec2)  # 计算叉积

    v_s = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)  # 计算叉积的模
    v_c = np.dot(vec1, vec2)  # 计算点积
    skew = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])  # 创建叉积的斜对称矩阵
    rotmat = np.eye(3) + skew + skew @ skew * ((1 - v_c) / (v_s ** 2))  # 计算旋转矩阵
    return rotmat  # 返回旋转矩阵


def get_stickhand(hand_arr, conn=None, color=[0.4, 0.4, 0.4]):  # 定义获取棍状手模型的函数
    conn = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [5, 6], [6, 7], [7, 8], [8, 9],
            [4, 10], [10, 11], [11, 12], [12, 13], [3, 14], [14, 15], [15, 16], [16, 17],
            [2, 18], [18, 19], [19, 20], [20, 21], [1, 22], [22, 23], [23, 24]]  # 定义连接关系

    def get_sphere(position, radius, color):  # 内部函数：创建球体
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius).paint_uniform_color(color)  # 创建并着色球体
        sp.translate(position)  # 平移到指定位置
        sp.compute_vertex_normals()  # 计算顶点法线
        return sp  # 返回球体

    def get_segment(parent, child, radius, color):  # 内部函数：创建连接段
        v = parent - child  # 计算方向向量
        seg = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(v), resolution=20,
                                                        split=1).paint_uniform_color(color)  # 创建并着色圆柱
        mat = getRotation(vec1=np.array([0, 0, 1]), vec2=v / np.linalg.norm(v))  # 计算旋转矩阵
        seg.rotate(mat)  # 应用旋转
        seg.translate((parent + child) / 2)  # 平移到中点
        seg.compute_vertex_normals()  # 计算顶点法线
        return seg  # 返回连接段

    mesh = o3d.geometry.TriangleMesh()  # 创建空三角网格
    for i in range(hand_arr.shape[0]):  # 遍历手部关节
        mesh += get_sphere(hand_arr[i], 0.003, [0.5, 0, 0.5])  # 添加关节球体
    for pairs in conn:  # 遍历连接关系
        p, c = pairs[0], pairs[1]  # 获取父子关节索引
        mesh += get_segment(hand_arr[p], hand_arr[c], 0.005, color)  # 添加连接段
    return mesh  # 返回手部网格


def get_stickman(body_position_arr, head_tip=None, color=[0.4, 0.4, 0.4], foot_contact=None):  # 定义获取棍状人体模型的函数
    body_line_idxs_int = [[0, 1], [0, 15], [0, 19], [1, 2], [2, 4], [4, 5], [4, 7], [4, 11], [5, 6],
                          [7, 8], [8, 9], [9, 10], [11, 12], [12, 13], [13, 14], [15, 16], [16, 17],
                          [17, 18], [19, 20], [20, 21], [21, 22]]  # 定义身体连接关系

    def get_sphere(position, radius, color):  # 内部函数：创建球体
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius).paint_uniform_color(
            [1, 0, 1])  # in rgb 0~1  # 创建并着色球体
        sp.translate(position)  # 平移到指定位置
        sp.compute_vertex_normals()  # 计算顶点法线
        return sp  # 返回球体

    def get_segment(parent, child, radius, color):  # 内部函数：创建连接段
        v = parent - child  # 计算方向向量
        seg = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(v), resolution=20,
                                                        split=1).paint_uniform_color(color)  # 创建并着色圆柱
        mat = getRotation(vec1=np.array([0, 0, 1]), vec2=v / np.linalg.norm(v))  # 计算旋转矩阵
        seg.rotate(mat)  # 应用旋转
        seg.translate((parent + child) / 2)  # 平移到中点
        seg.compute_vertex_normals()  # 计算顶点法线
        return seg  # 返回连接段

    def get_foot(parent, child, color, width=0.05):  # 内部函数：创建脚部模型
        v = parent - child  # 计算方向向量
        height = np.linalg.norm(v)  # 计算高度
        depth = width / 2  # 计算深度
        mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth).paint_uniform_color(
            color)  # 创建并着色长方体
        mesh.translate([-width / 2, -height / 2, -depth / 2])  # 平移使中心对齐
        mat = getRotation(vec1=np.array([0, 1, 0]), vec2=v / height)  # 计算旋转矩阵
        mesh.rotate(mat)  # 应用旋转
        mesh.translate((parent + child) / 2)  # 平移到中点
        mesh.compute_vertex_normals()  # 计算顶点法线
        return mesh  # 返回脚部模型

    body_mesh = o3d.geometry.TriangleMesh()  # 创建空三角网格
    for bidx in range(23):  # 遍历身体关节
        if bidx == 6 and head_tip is not None:  # jC1Head  # 如果是头部关节且有头顶点
            body_mesh += get_segment(body_position_arr[6], head_tip, 0.08, color)  # 添加头部连接段
        elif bidx in [12, 8, 15, 19]:  # 'jLeftShoulder','jRightShoulder','jRightHip','jLeftHip'  # 如果是肩部或髋部关节
            body_mesh += get_sphere(body_position_arr[bidx], 0.03, color)  # 添加较大的关节球体
        elif bidx == 22:  # "jLeftBallFoot":  # 如果是左脚掌
            if foot_contact is not None:  # 如果有足部接触信息
                if foot_contact[22]:  # 如果左脚接触地面
                    body_mesh += get_sphere(body_position_arr[bidx], 0.03, [1, 0, 0])  # 添加红色关节球体
                else:  # 如果左脚不接触地面
                    body_mesh += get_sphere(body_position_arr[bidx], 0.03, color)  # 添加正常颜色的关节球体

        elif bidx == 18:  # "jRightBallFoot":  # 如果是右脚掌
            if foot_contact is not None:  # 如果有足部接触信息
                if foot_contact[18]:  # 如果右脚接触地面
                    body_mesh += get_sphere(body_position_arr[bidx], 0.03, [1, 0, 0])  # 添加红色关节球体
                else:  # 如果右脚不接触地面
                    body_mesh += get_sphere(body_position_arr[bidx], 0.03, color)  # 添加正常颜色的关节球体

        else:  # 其他关节
            body_mesh += get_sphere(body_position_arr[bidx], 0.02, color)  # 添加标准大小的关节球体

    for pidx, cidx in body_line_idxs_int:  # 遍历身体连接关系
        parent, child = body_position_arr[pidx], body_position_arr[cidx]  # 获取父子关节位置
        if pidx == 21 or pidx == 17:  # 如果是脚踝到脚掌的连接
            body_mesh += get_foot(parent, child, width=0.05, color=color)  # 添加脚部模型
        else:  # 其他连接
            body_mesh += get_segment(parent, child, 0.02, color)  # 添加标准连接段
    return body_mesh  # 返回身体网格


class simpleViewer(object):  # 定义简单查看器类
    def __init__(self, title, width, height, view_set_list, view=None):  # 初始化方法
        app = o3d.visualization.gui.Application.instance  # 获取应用实例
        app.initialize()  # 初始化应用
        self.main_vis = o3d.visualization.O3DVisualizer(title, width, height)  # 创建可视化器
        self.main_vis.show_settings = False  # 不显示设置面板
        self.main_vis.show_skybox(False)  # 不显示天空盒
        app.add_window(self.main_vis)  # 添加窗口到应用

        if view is not None:  # 如果提供了视图
            self.intrinsic = view.intrinsic  # 设置内参

    def export_view(self):  # 导出视图方法
        return self.curview  # 返回当前视图

    def setupcamera(self, extrinsic_matrix):  # 设置相机方法
        self.main_vis.setup_camera(self.intrinsic, extrinsic_matrix)  # 设置相机参数

    def tick(self):  # 更新方法
        app = o3d.visualization.gui.Application.instance  # 获取应用实例
        tick_return = app.run_one_tick()  # 运行一次更新
        if tick_return:  # 如果更新成功
            self.main_vis.post_redraw()  # 触发重绘
        return tick_return  # 返回更新结果

    def add_plane(self, resolution=128, bound=100, up_vec='z'):  # 添加平面方法
        def makeGridPlane(bound=100., resolution=128, color=np.array([0.5, 0.5, 0.5]), up='z'):  # 内部函数：创建网格平面
            min_bound = np.array([-bound, -bound])  # 设置最小边界
            max_bound = np.array([bound, bound])  # 设置最大边界
            xy_range = np.linspace(min_bound, max_bound, num=resolution)  # 创建线性空间
            grid_points = np.stack(np.meshgrid(*xy_range.T), axis=-1).astype(np.float32)  # asd  # 创建网格点
            if up == 'z':  # 如果上方向是z轴
                grid3d = np.concatenate(
                    [grid_points, np.zeros_like(grid_points[:, :, 0]).reshape(resolution, resolution, 1)],
                    axis=2)  # 创建3D网格，z坐标为0
            elif up == 'y':  # 如果上方向是y轴
                grid3d = np.concatenate([grid_points[:, :, 0][:, :, None],
                                         np.zeros_like(grid_points[:, :, 0]).reshape(resolution, resolution, 1),
                                         grid_points[:, :, 1][:, :, None]], axis=2)  # 创建3D网格，y坐标为0
            elif up == 'x':  # 如果上方向是x轴
                grid3d = np.concatenate(
                    [np.zeros_like(grid_points[:, :, 0]).reshape(resolution, resolution, 1), grid_points],
                    axis=2)  # 创建3D网格，x坐标为0
            else:  # 如果上方向未指定
                print("Up vector not specified")  # 打印警告
                return None  # 返回空
            grid3d = grid3d.reshape((resolution ** 2, 3))  # 重塑为点列表
            indices = []  # 初始化索引列表
            for y in range(resolution):  # 遍历y方向
                for x in range(resolution):  # 遍历x方向
                    corner_idx = resolution * y + x  # 计算当前点索引
                    if x + 1 < resolution:  # 如果x方向有下一个点
                        indices.append((corner_idx, corner_idx + 1))  # 添加水平线段
                    if y + 1 < resolution:  # 如果y方向有下一个点
                        indices.append((corner_idx, corner_idx + resolution))  # 添加垂直线段

            line_set = o3d.geometry.LineSet(  # 创建线集
                points=o3d.utility.Vector3dVector(grid3d),  # 设置点
                lines=o3d.utility.Vector2iVector(indices),  # 设置线
            )
            # line_set.colors = o3d.utility.Vector3dVector(colors)
            line_set.paint_uniform_color(color)  # 设置颜色

            return line_set  # 返回线集

        plane = makeGridPlane(bound, resolution, up=up_vec)  # 创建网格平面
        self.main_vis.add_geometry({"name": "floor", "geometry": plane})  # 添加平面到可视化器
        return  # 返回

    def remove_plane(self):  # 移除平面方法
        self.main_vis.remove_geometry({"name": "floor"})  # 从可视化器中移除平面
        return  # 返回

    def add_geometry(self, geometry: dict):  # 添加几何体方法
        self.main_vis.add_geometry(geometry)  # 添加几何体到可视化器

    def write_image(self, imagepath):  # 写入图像方法
        self.main_vis.export_current_image(imagepath)  # 导出当前图像

    def transform(self, name, transform_mtx):  # 变换几何体方法
        self.main_vis.scene.set_geometry_transform(name, transform_mtx)  # 设置几何体的变换矩阵

    def set_background(self, image):  # 设置背景方法
        self.main_vis.set_background([1, 1, 1, 0], image)  # 设置背景图像

    def remove_geometry(self, geom_name):  # 移除几何体方法
        self.main_vis.remove_geometry(geom_name)  # 从可视化器中移除几何体

    def run(self):  # 运行方法
        app = o3d.visualization.gui.Application.instance  # 获取应用实例
        app.run()  # 运行应用