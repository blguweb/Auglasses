# import numpy as np

# def normalize(v):
#     """ 标准化向量 """
#     norm = np.linalg.norm(v)
#     return v / norm if norm > 0 else v

# def rotation_matrix_to_align_with_x_axis(vector):
#     """ 创建一个旋转矩阵，使得 vector 旋转到与 x 轴对齐 """
#     if np.all(vector == 0):
#         return np.eye(3)  # 如果向量是零向量，则返回单位矩阵

#     # 目标向量是 x 轴
#     target = np.array([0, 1, 0])

#     # 标准化原始向量
#     vector_normalized = normalize(vector)

#     # 计算旋转轴（叉积）
#     v = np.cross(vector_normalized, target)

#     # 计算需要旋转的角度（点积）
#     c = np.dot(vector_normalized, target)

#     # 构建旋转矩阵
#     vx, vy, vz = v
#     s = np.linalg.norm(v)
#     kmat = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
#     rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

#     return rotation_matrix

# def map_vector_to_x_axis(vector):
#     """ 将向量映射到 x 轴 """
#     rot_matrix = rotation_matrix_to_align_with_x_axis(vector)
#     mapped_vector = rot_matrix.dot(vector)
#     return mapped_vector

# # 示例向量
# vector = np.array([1, 2, 3])

# # 映射到 x 轴
# mapped_vector = map_vector_to_x_axis(vector)
# print("Original Vector:", vector)
# print("Mapped Vector:", mapped_vector)
#######################################################
# import numpy as np

# def rotation_matrix_to_align_with_y_axis(x, y, z):
#     # 计算将点投影到xz平面所需的绕y轴旋转角度
#     theta_z = np.arctan2(z, x)

#     # 旋转矩阵绕y轴
#     Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
#                    [np.sin(theta_z), np.cos(theta_z), 0],
#                    [0, 0, 1]])

#     # 应用旋转后点的新坐标
#     point_rotated_z = np.dot(Rz, np.array([x, y, z]))
#     print(point_rotated_z)
#     # 计算绕新x轴旋转的角度
#     theta_x = np.arctan2(point_rotated_z[2], point_rotated_z[1])

#     # 旋转矩阵绕新x轴
#     Rx = np.array([[1, 0, 0],
#                    [0, np.cos(theta_x), -np.sin(theta_x)],
#                    [0, np.sin(theta_x), np.cos(theta_x)]])

#     # 最终旋转矩阵是Rz和Rx的乘积
#     R = np.dot(Rx, Rz)

#     return R

# # 已知点的坐标
# x, y, z = 3, 4, 5

# # 获取旋转矩阵
# R = rotation_matrix_to_align_with_y_axis(x, y, z)
# R


# import numpy as np
# def standardized_coordinate_system_rotation(x, y, z):
#     """
#     Compute a standardized rotation of the coordinate system such that any point (x, y, z)
#     is aligned along the new y-axis (y') in the rotated coordinate system, while keeping the
#     orientation of x' and z' axes consistent across different points.
#     """
#     # Distance from the point to the origin
#     distance = np.sqrt(x**2 + y**2 + z**2)

#     # Compute the angles for standardized rotation
#     # Angle to rotate around z-axis
#     theta_z = np.arctan2(z, x)

#     # Rotate the point onto the xz-plane
#     xz_projection = np.sqrt(x**2 + z**2)

#     # Angle to rotate around y-axis
#     theta_y = np.arctan2(xz_projection, y)

#     # Rotation matrix around z-axis
#     Rz = np.array([[np.cos(theta_z), 0, np.sin(theta_z)],
#                    [0, 1, 0],
#                    [-np.sin(theta_z), 0, np.cos(theta_z)]])

#     # Rotation matrix around y-axis
#     Ry = np.array([[np.cos(theta_y), -np.sin(theta_y), 0],
#                    [np.sin(theta_y), np.cos(theta_y), 0],
#                    [0, 0, 1]])

#     # Combined rotation matrix
#     R = np.dot(Ry, Rz)

#     return R, distance

# # Calculate the standardized coordinate system rotation
# x = 0
# y = 0
# z = 1
# R_standardized_coord_system, distance_standardized_coord_system = standardized_coordinate_system_rotation(x, y, z)
# print(R_standardized_coord_system)
# result = np.dot(R_standardized_coord_system, np.array([0,0,-1]))
# print(result,np.sqrt(x**2 + y**2 + z**2))


import numpy as np
import matplotlib.pyplot as plt

def visual_pred_lens(pred_lens, output_channels, true=None, preds=None,  name='./pic/test.pdf'):
    """
    Results visualization
    """

    rows = 4  # 选择适当的行数
    cols = output_channels // rows + (output_channels % rows > 0)

    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))  # 调整大小以适应所有子图
    axs = axs.flatten()  # 将多维数组展平，便于索引
    x = np.arange(0, pred_lens)
    for i in range(output_channels):
        axs[i].plot(x, preds[:, i], label='Prediction', linewidth=2)  # 绘制模型输出
        axs[i].plot(x, true[:, i], label='GroundTruth', linewidth=2)  # 绘制实际值
        axs[i].set_title(f'Channel {i+1}')  # 设置子图标题
        axs[i].legend()  # 显示图例

    # 对于不需要的子图位置，关闭它们
    for i in range(output_channels, rows*cols):
        fig.delaxes(axs[i])
    plt.legend()
    plt.tight_layout()  # 调整子图间距
    plt.savefig(name, bbox_inches='tight')


gt=np.random.random((48,17))
pd=np.random.random((48,17))
visual_pred_lens(48, 17, true=gt, preds=pd,  name='./test.png')