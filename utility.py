import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, lfilter
from datetime import datetime
import matplotlib.pyplot as plt
"""
@Description :  通用的函数
@Author      :  kidominox 
@Time        :   2023/12/13 14:31:54
"""
def unit_conversion(raw_data):
    raw_data[:,0] = (raw_data[:,0] * 9.8) / 16384
    raw_data[:,1] = (raw_data[:,1] * 9.8) / 16384
    raw_data[:,2] = (raw_data[:,2] * 9.8) / 16384
    raw_data[:,3] = (raw_data[:,3] * 2000) / 0x8000
    raw_data[:,4] = (raw_data[:,4] * 2000) / 0x8000
    raw_data[:,5] = (raw_data[:,5] * 2000) / 0x8000
    return raw_data

def find_index(time,imu_time):
    imu_i = 0
    while imu_i <= len(imu_time):
        if imu_time[imu_i] >= time:
            break
        imu_i += 1
    return imu_i

# Function for low-pass filtering
def low_pass_filter(data, cutoff=20, fs=400, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    # plt.plot(data)
    # plt.plot(filtered_data)
    # plt.show()
    return filtered_data


def mappingMethod(A,B):
    # A -> B
    if A.shape != B.shape:
        raise ValueError("Input matrices A and B must have the same shape")
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    N = A.shape[0]
    
    A = A - np.tile(centroid_A, (N, 1))
    B = B - np.tile(centroid_B, (N, 1))
    A = np.float64(A)
    B = np.float64(B)
    H = np.dot(A.T, B)
    
    U, S, Vt = np.linalg.svd(H)
    
    R = np.dot(Vt.T, U.T)
    
    
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] = -Vt[2, :]
        R = np.dot(Vt.T, U.T)
    
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    
    return R, t

def mappingUniform(data,index_data,time_data,imu_time):
    # 求出静止的片段；然后截取出来
    start_index, end_index = 0,0
    for i in range(3):
        # 取第一个epoch的静止片段进行映射，求出参数
        start_index= find_index(time_data[0,i*2],imu_time)
        end_index = find_index(time_data[0,i*2+1],imu_time)
        length = end_index - start_index
        var_ = np.var(data[start_index:start_index + int(length / 2),4])
        if var_ < 1.1:
            break
        else:
            print("var of static state is too large!",var_)
    # 静止片段
    left_data =data[start_index:end_index,0:6]
    right_data = data[start_index:end_index,6:12]
    calibration_data = data[start_index:end_index,12:18]


    left_data = np.float64(left_data)
    right_data = np.float64(right_data)
    calibration_data = np.float64(calibration_data)

    N = end_index - start_index
    # print(type(left_data[0,0]))
    R_left,T_left = mappingMethod(calibration_data,left_data)
    R_right,T_right = mappingMethod(calibration_data,right_data)


    left_calibration = np.dot(R_left, data[:,12:18].T) + np.tile(T_left.reshape(-1, 1), (1, data.shape[0]))
    right_calibration = np.dot(R_right, data[:,12:18].T) + np.tile(T_right.reshape(-1, 1), (1, data.shape[0]))
    left_calibration = np.transpose(left_calibration)
    right_calibration = np.transpose(right_calibration)

    # # left - calibration
    data[:,0:6] = data[:,0:6] - left_calibration
    data[:,6:12] = data[:,6:12] - right_calibration

    return data, R_left, T_left, R_right, T_right



def remove_imu_peak(imu_data):
    # data (length,imu category)
    for i in range(imu_data.shape[1]):
        if i in [0,1,2,6,7,8,12,13,14]:
            theta = 4
        else:
            theta =12
        index = ((imu_data[:,i] - np.roll(imu_data[:,i],1) > theta) & (imu_data[:,i] - np.roll(imu_data[:,i],-1) > theta))\
              | ((imu_data[:,i] - np.roll(imu_data[:,i],1) < -theta) & (imu_data[:,i] - np.roll(imu_data[:,i],-1) < -theta))
        # plt.plot(imu_data[:,i])
        # plt.plot(index.nonzero()[0], imu_data[index,i], 'ro')
        # plt.show()
        index = np.nonzero(index)[0]
        for j in index:
            if j == imu_data.shape[0]-1:
                imu_data[j,i] = imu_data[j-1,i]
            else:
                imu_data[j,i] = (imu_data[j+1,i] +imu_data[j-1,i]) / 2
    return imu_data

def normalize(v):
    """ 标准化向量 """
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def rotation_matrix_to_align_with_x_axis(vector):
    # 旋转点
    # 创建一个旋转矩阵，使得 vector 旋转到与 x 轴对齐
    # 旋转通常是最短路径旋转，即旋转角度最小的那种。因为imu变化不大，所以不会有另外一种旋转结果
    if np.all(vector == 0):
        return np.eye(3)  # 如果向量是零向量，则返回单位矩阵

    # 目标向量是 y 轴
    target = np.array([0, -1, 0])

    # 标准化原始向量
    vector_normalized = normalize(vector)

    # 计算旋转轴（叉积）
    v = np.cross(vector_normalized, target)

    # 计算需要旋转的角度（点积）y
    c = np.dot(vector_normalized, target)
    print("c",c)

    # 构建旋转矩阵
    vx, vy, vz = v
    s = np.linalg.norm(v)
    kmat = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix


def standardized_coordinate_system_rotation(x, y, z):
    """
    Compute a standardized rotation of the coordinate system such that any point (x, y, z)
    is aligned along the new y-axis (y') in the rotated coordinate system, while keeping the
    orientation of x' and z' axes consistent across different points.
    """
    # 旋转坐标系
    # Distance from the point to the origin
    distance = np.sqrt(x**2 + y**2 + z**2)

    # Compute the angles for standardized rotation
    # Angle to rotate around z-axis
    theta_z = np.arctan2(z, x)

    # Rotate the point onto the xz-plane
    xz_projection = np.sqrt(x**2 + z**2)

    # Angle to rotate around y-axis
    theta_y = np.arctan2(xz_projection, y)

    # Rotation matrix around z-axis
    Rz = np.array([[np.cos(theta_z), 0, np.sin(theta_z)],
                   [0, 1, 0],
                   [-np.sin(theta_z), 0, np.cos(theta_z)]])

    # Rotation matrix around y-axis
    Ry = np.array([[np.cos(theta_y), -np.sin(theta_y), 0],
                   [np.sin(theta_y), np.cos(theta_y), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(Ry, Rz)

    return R

def static_xyz(imu_data, index_data, time_data, imu_time, epoch):
    for i in range(len(index_data)):
        if index_data[i] == 0:
            acc_mean_m = []
            for j in range(epoch):
                start_time = time_data[i, j*2]
                end_time = time_data[i, j*2+1]
                # print(start_time,end_time)
                start_index = find_index(start_time,imu_time)-200
                end_index = find_index(end_time,imu_time)
                acc_mean_m.append([
                        np.mean(low_pass_filter(imu_data[start_index:end_index, k]))
                        for k in [6, 7, 8]
                    ])
            acc_mean = np.mean(acc_mean_m, axis=0)
            # 方差
            acc_var = np.var(acc_mean_m, axis=0)
            print("acc_mean", acc_mean)
            print("acc_var", acc_var)
            if np.all(acc_var < 0.02):
                return acc_mean
            else:
                print("var of static state is too large!")
                return None

def filterData(imu_data,cutoff = 20):
    filter_data = np.zeros_like(imu_data)
    for j in range(imu_data.shape[1]):
        filter_data[:,j] = low_pass_filter(imu_data[:,j], cutoff= cutoff)
    return filter_data

def calibration(imu_data, index_data, time_data, imu_time, epoch):
    # 创建 imu_data 的副本
    imu_data_copy = imu_data.copy()
    # 提取静止状态；
    acc = static_xyz(imu_data_copy, index_data, time_data, imu_time, epoch)
    if acc is not None:
        # 将向量映射到 x 轴
        # rot_matrix = rotation_matrix_to_align_with_x_axis(acc)
        rot_matrix = standardized_coordinate_system_rotation(acc[0], acc[1], acc[2])
        # print(rot_matrix)                    
        # right
        imu_data_copy[:,6:9] = rot_matrix.dot(imu_data_copy[:,6:9].T).T
        imu_data_copy[:,9:12] = rot_matrix.dot(imu_data_copy[:,9:12].T).T
        return imu_data_copy
    else:
        print("Calibration failed!")
        return None

def calculate_frame_range_from_timestamps(imu_start_timestamp, imu_end_timestamp, video_start_timestamp, frame_start, openface_data, frame_rate=30):
    """
    Calculate the corresponding frame range in the video for a given IMU data segment based on timestamps.

    :param imu_start_timestamp: The start timestamp of the IMU segment (in 'HH:MM:SS.fff' format).
    :param imu_end_timestamp: The end timestamp of the IMU segment (in 'HH:MM:SS.fff' format).
    :param video_start_timestamp: The timestamp of the first IMU data point corresponding to the video start frame.
    :param frame_start: The frame number in the video corresponding to the first IMU data point.
    :param frame_rate: The frame rate of the video (frames per second).
    :return: A tuple (start_frame, end_frame) indicating the range of frames in the video.
    """
    # Convert timestamps to datetime objects
    imu_start_dt = datetime.strptime(imu_start_timestamp, '%H:%M:%S.%f')
    imu_end_dt = datetime.strptime(imu_end_timestamp, '%H:%M:%S.%f')
    video_start_dt = datetime.strptime(video_start_timestamp, '%H:%M:%S.%f')

    # Calculate time differences in seconds
    start_diff = (imu_start_dt - video_start_dt).total_seconds()
    end_diff = (imu_end_dt - video_start_dt).total_seconds()
    

    # # Calculate the corresponding frame numbers
    # start_frame = frame_start + int(start_diff * frame_rate)
    # end_frame = frame_start + int(end_diff * frame_rate)

    # Find the closest timestamps in the openface_data
    start_frame = np.abs(openface_data[:,0] - openface_data[frame_start,0] - start_diff).argmin()
    end_frame = np.abs(openface_data[:,0] - openface_data[frame_start,0] - end_diff).argmin()

    print(start_frame,end_frame)

    return start_frame, end_frame


''' 
    @description:  提取最大值进行归一化
'''
def calculate_maximum(imu_data, index_data, time_data, imu_time, first_frame, openface_data,epoch = 3):
    epoch_maximum = []
    for ex in range(len(index_data)):
        # expression_list[index_data[ex]] : category
        # 用于存储每个 epoch 计算结果的临时列表
        if ex == 5: # open mouth
            for ep in range(epoch):
                # identified imu part
                start_time = time_data[ex,ep*2]
                end_time = time_data[ex,ep*2+1]
                start_index = find_index(start_time,imu_time)
                end_index = find_index(end_time,imu_time)
                # 片段低通滤波
                filtered_signal = filterData(imu_data[start_index:end_index,:])
                # 计算对应的视频帧范围
                start_frame, end_frame = calculate_frame_range_from_timestamps(start_time, end_time, imu_time[0], first_frame, openface_data)

                # 提取最大值进行归一化
                imu_acc_max = np.max(abs(filtered_signal[:,6:9]))

                imu_acc_max_left = np.max(abs(filtered_signal[:,0:3]))
                print("imu_acc_max_left",imu_acc_max_left,"imu_acc_max",imu_acc_max)

                # au max
                # 从所有曲线中提取最大值
                au_max = np.max(openface_data[start_frame:end_frame, 1:])
                maximum = imu_acc_max / (au_max / 5)
                epoch_maximum.append(maximum)
    return np.max(epoch_maximum)


'''
    @description:  归一化
'''
def normalization(signal, acc_max = 1.2, gyto_max = 80):
    acc_min = -acc_max
    gyto_min = -gyto_max
    # acc
    signal[:,6:9] = 2 * ((signal[:,6:9] - acc_min) / (acc_max - acc_min)) - 1
    signal[:,0:3] = 2 * ((signal[:,0:3] - acc_min) / (acc_max - acc_min)) - 1  
    
    # gyto
    signal[:,9:12] = 2 * ((signal[:,9:12] - gyto_min) / (gyto_max - gyto_min)) - 1
    signal[:,3:6] = 2 * ((signal[:,3:6] - gyto_min) / (gyto_max - gyto_min)) - 1
    return signal


'''
    @description: 提取信号强度超过一定阈值时长的片段
'''
def calculate_expression_duration(signal, threshold):
    """
    Calculate the duration where each of the three columns in the signal independently exceeds a certain threshold.
    Extracts the earliest start and latest end index across these columns.

    :param signal: Array of signal values [num_samples, num_axes]
    :param threshold: Threshold value to identify expression
    :return: Sub-array of the signal from the earliest start to the latest end index
    """
    start_indices = []
    end_indices = []
    offset = 50

    for col in range(9, 12):  # left and right are the same
        column_data = abs(signal[:, col])
        above_threshold = column_data > threshold

        # Find first index exceeding the threshold
        start_idx = np.argmax(above_threshold) if np.any(above_threshold) else None

        # Find last index exceeding the threshold
        end_idx = len(column_data) - np.argmax(above_threshold[::-1]) - 1 if np.any(above_threshold) else None
        

        if start_idx is not None:
            start_indices.append(start_idx)
        if end_idx is not None:
            end_indices.append(end_idx)

    # Find the earliest start and latest end index across all columns
    if start_indices and end_indices:
        earliest_start = min(start_indices)
        latest_end = max(end_indices)
        # return signal[earliest_start : latest_end + 1, :]
        if earliest_start - offset >= 0:
            earliest_start -= offset
        else:
            earliest_start = 0
        if latest_end + offset < signal.shape[0]:
            latest_end += offset
        else:
            latest_end = signal.shape[0] - 1
        print("start to end ",earliest_start,latest_end)

        return earliest_start, latest_end, signal[earliest_start : latest_end + offset, :]
    else:
        print("No signal exceeded the threshold.")
        return None
    
def correlationCoefficient(X, Y,label):
    # acc 

    # 计算每个维度的相关系数
    # right_imu_data = np.float64(right_imu_data)
    # reference_imu_data = np.float64(reference_imu_data)
    if label == "right":
        x = X[:, 6:12]
        y = Y[:, 6:12]
        epoch_correlations = [np.max(np.correlate(x[:, i], y[:, i],mode='full')) for i in range(6)]
    elif label == "left":
        x = X[:, 0:6]
        y = Y[:, 0:6]
        epoch_correlations = [np.max(np.correlate(x[:, i], y[:, i],mode='full')) for i in range(6)]
    return epoch_correlations


def correlationsJudge(avg_left_correlations, avg_right_correlations, expression):
    # print("all",avg_left_correlations,avg_right_correlations)
    if expression == 'happy':
        if avg_left_correlations[0] < 20:
            print("left imu failed to collect happy correctly!", avg_left_correlations)
        if avg_right_correlations[0] < 20:
            print("right imu failed to collect happy correctly!", avg_right_correlations)
    elif expression == 'frown':
        if avg_left_correlations[2] < 20:
            print("left imu failed to collect frown correctly!", avg_left_correlations)
        if avg_right_correlations[2] < 20:
            print("right imu failed to collect frown correctly!", avg_right_correlations)
    elif expression == 'openmouth':
        if avg_left_correlations[2] < 20:
            print("left imu failed to collect openmouth correctly!", avg_left_correlations)
        if avg_right_correlations[2] < 20:
            print("right imu failed to collect openmouth correctly!", avg_right_correlations)
