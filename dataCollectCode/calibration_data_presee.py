import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, lfilter
import matplotlib.pyplot as plt
import os
"""
@Description :  判断镜腿imu信号和凸起处的imu信号的相关性，采用信号的绘制和相关度分析
@Author      :  kidominox 
@Time        :   2023/12/01 22:50:41
"""
ex_per = 'pre_yl3'
imu_path = "./dataCollection/imu_" + ex_per + ".csv" 
time_path = "./dataCollection/time_" + ex_per + ".csv"
index_path = "./dataCollection/index_" + ex_per + ".csv"

feature = ["imu_left_accx","imu_left_accy","imu_left_accz","imu_left_gyrox","imu_left_gyroy","imu_left_gyroz",\
    "imu_right_accx","imu_right_accy","imu_right_accz","imu_right_gyrox","imu_right_gyroy","imu_right_gyroz",\
    "imu_head_accx","imu_head_accy","imu_head_accz","imu_head_gyrox","imu_head_gyroy","imu_head_gyroz"]
expression_list = ["none","happy","frown","openmouth"]
# expression_list = ["none","happy","upfrown","frown","open mouth"]
plot_title = ["accx","accy","accz","gyrox","gyroy","gyroz"]
def removeImuPeak(imu_data):
    # data (length,imu category)
    for i in range(imu_data.shape[1]):
        if i in [0,1,2,6,7,8,12,13,14]:
            theta = 4
        else:
            theta =12
        index = ((imu_data[:,i] - np.roll(imu_data[:,i],1) > theta) & (imu_data[:,i] - np.roll(imu_data[:,i],-1) > theta)) | ((imu_data[:,i] - np.roll(imu_data[:,i],1) < -theta) & (imu_data[:,i] - np.roll(imu_data[:,i],-1) < -theta))
        # plt.plot(imu_data[:,i])
        # plt.plot(index.nonzero()[0], imu_data[index,i], 'ro')
        # plt.show()
        index = np.nonzero(index)
        for j in index:
            imu_data[j,i] = (imu_data[j+1,i] +imu_data[j-1,i]) / 2
    return imu_data

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

def mappingUniform(data,index_data,time_data):
    # 求出静止的片段；然后截取出来
    start_index, end_index = 0,0
    for i in range(len(expression_list)):
        if index_data[i] == 0:
            # 取第一个epoch的静止片段进行映射，求出参数
            start_index= find_index(time_data[i,2],imu_time) - 200
            end_index = find_index(time_data[i,3],imu_time)
            break
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

    # DATA: [imu_left,imu_right,imu_left_calibration,imu_right_calibration]
    # data[:,12:18] = left_calibration
    # data = np.hstack((data[:,:],right_calibration))

    # print("mapping uniform done",data.shape)
    return data

# Function for low-pass filtering
def low_pass_filter(data, cutoff=10, fs=400, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    # plt.plot(data)
    # plt.plot(filtered_data)
    # plt.show()
    return filtered_data

def unitConversion(raw_data):
    raw_data[:,0] = (raw_data[:,0] * 9.8) / 16384
    raw_data[:,1] = (raw_data[:,1] * 9.8) / 16384
    raw_data[:,2] = (raw_data[:,2] * 9.8) / 16384
    raw_data[:,3] = (raw_data[:,3] * 2000) / 0x8000
    raw_data[:,4] = (raw_data[:,4] * 2000) / 0x8000
    raw_data[:,5] = (raw_data[:,5] * 2000) / 0x8000
    return raw_data

def signalPlot(selected_data, expression, number):
    # 右侧 IMU 和参考 IMU 的数据
    left_imu_data = selected_data[:, 0:6]
    right_imu_data = selected_data[:, 6:12]
    # 绘制图像，共 4x3 个子图
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # 设置标题
    fig.suptitle('Right IMU vs Left IMU Signal('+ expression + '_' + str(number) + ')', fontsize=16)
    

    # 遍历每个子图并绘制相应的数据
    for i in range(2):
        for j in range(3):
            # 计算子图索引
            index = i*3 + j

            # 从右侧 IMU 数据中获取信号
            right_signal = right_imu_data[:, index]
            # 从参考 IMU 数据中获取信号
            left_signal = left_imu_data[:, index]

            # 绘制右侧 IMU 信号
            axs[i, j].plot(right_signal, label='Right IMU', linestyle='-')
            # 绘制参考 IMU 信号
            axs[i, j].plot(left_signal, label='Left IMU', linestyle='--')
            # 设置图例
            axs[i, j].legend()
            # 设置子图标题
            axs[i, j].set_title(f'Signal of {plot_title[index]}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局
    # plt.show()  # 显示图像
    
    # 如果没有文件夹新建
    
    if not os.path.exists('./dataset_presee/' + ex_per):
        os.makedirs('./dataset_presee/' + ex_per)
    # 保存图像到文件
    file_path = './dataset_presee/' + ex_per + '/IMU('+ expression + '_' + str(number) + ').png'
    fig.savefig(file_path)
    
    plt.cla()
    plt.close("all")

# 计算映射后的参考imu和感知imu之间的相关性
def correlationCoefficient(selected_data):
    # 返回segment data 的每一个维度的相关系数
    # 右侧 IMU 和参考 IMU 的数据
    right_imu_data = selected_data[:, 6:12]
    reference_imu_data = selected_data[:, 18:24]
    # 计算每个维度的相关系数
    right_imu_data = np.float64(right_imu_data)
    reference_imu_data = np.float64(reference_imu_data)
    epoch_correlations = [np.cov(right_imu_data[:, i], reference_imu_data[:, i])[0, 1] for i in range(6)]

    # epoch_correlations = [np.corrcoef(right_imu_data[:, i], reference_imu_data[:, i])[0, 1] for i in range(6)]
    # epoch_correlations = []
    # for i in range(6):
    #     right_signal = right_imu_data[:, i]
    #     reference_signal = reference_imu_data[:, i]

    #     # 检查 NaN 值和无限值
    #     if np.isnan(right_signal).any() or np.isnan(reference_signal).any():
    #         print(f"Warning: NaN values found in signals for dimension {i}")
    #         continue
    #     if np.isinf(right_signal).any() or np.isinf(reference_signal).any():
    #         print(f"Warning: Infinite values found in signals for dimension {i}")
    #         continue

    #     # 检查单一值数组
    #     if np.std(right_signal) == 0 or np.std(reference_signal) == 0:
    #         print(f"Warning: No variation in one of the signals for dimension {i}")
    #         continue

    #     # 计算相关系数
    #     corr_coeff = np.corrcoef(right_signal, reference_signal)[0, 1]
    #     epoch_correlations.append(corr_coeff)
    return epoch_correlations

def filterData(imu_data):
    filter_data = np.zeros_like(imu_data)
    for j in range(imu_data.shape[1]):
        filter_data[:,j] = low_pass_filter(imu_data[:,j])
    return filter_data
    
def find_index(time,imu_time):
    imu_i = 0
    while imu_i <= len(imu_time):
        if imu_time[imu_i] >= time:
            break
        imu_i += 1
    return imu_i

if __name__ == "__main__":

    imu_raw_data = pd.read_csv(imu_path,header= None)
    imu_raw_data = imu_raw_data.values
    # [start time, end time] * epoch
    time_data = pd.read_csv(time_path,header= None)
    time_data = time_data.values
    # index: ["静止","开心","难过","害怕","生气","惊喜","厌恶","轻蔑"]的索引
    index_data = pd.read_csv(index_path,header= None)
    index_data = index_data.values
    index_data = index_data.flatten()

    imu_time = imu_raw_data[:,0]
    imu_data = imu_raw_data[:,2:]
    time_data = time_data[:,:]
    imu_data[:,0:6] = unitConversion(imu_data[:,0:6])
    imu_data[:,6:12] = unitConversion(imu_data[:,6:12])
    imu_data[:,12:18] = unitConversion(imu_data[:,12:18])
    # 去除峰异常值
    imu_data = removeImuPeak(imu_data)
    # 映射
    imu_data = mappingUniform(imu_data,index_data,time_data)
    filtered_signal = filterData(imu_data)

    epoch = 3

    # 初始化存储结果的 DataFrame
    correlation_results = pd.DataFrame(columns=['Expression', 'IMU_accx', 'IMU_accy', 'IMU_accz', 
                                            'IMU_gyrox', 'IMU_gyroy', 'IMU_gyroz'])
    # segment
    for ex in range(len(expression_list)):
        # expression_list[index_data[ex]] : category
        # 用于存储每个 epoch 计算结果的临时列表
        temp_correlations = []

        for ep in range(epoch):
            
            start_time = time_data[ex,ep*2]
            end_time = time_data[ex,ep*2+1]
            start_index = find_index(start_time,imu_time) - 200
            end_index = find_index(end_time,imu_time) + 800
            # # 片段低通滤波
            # filtered_signal = filterData(imu_data[start_index:end_index,:])
            # 绘制图像
            signalPlot(filtered_signal[start_index:end_index,:],expression_list[index_data[ex]],ep)

        #     # 计算相关系数
        #     epoch_correlations = correlationCoefficient(filtered_signal)
        #     temp_correlations.append(epoch_correlations)
        # # 计算每个维度的平均相关系数
        # avg_correlations = np.mean(temp_correlations, axis=0)
        # correlation_results = correlation_results.append({'Expression': expression_list[index_data[ex]], 
        #                                                 'IMU_accx': avg_correlations[0],
        #                                                 'IMU_accy': avg_correlations[1],
        #                                                 'IMU_accz': avg_correlations[2],
        #                                                 'IMU_gyrox': avg_correlations[3],
        #                                                 'IMU_gyroy': avg_correlations[4],
        #                                                 'IMU_gyroz': avg_correlations[5]}, 
        #                                                 ignore_index=True)
    # 将结果保存到 CSV 文件
    # file_path = './artifacts/cov_' + ex_per +'.csv'
    # correlation_results.to_csv(file_path, index=False)
            

