import numpy as np
import pandas as pd
from utility import *
import os
"""
@Description :  探究初始imu的位置不同对信号的影响
@Author      :  kidominox 
@Time        :   2023/12/14 12:34:45
"""

# 
ex_per = 'd_3_10_3'
imu_path = "./selectSize/imu_" + ex_per + ".csv" 
time_path = "./selectSize/time_" + ex_per + ".csv"
index_path = "./selectSize/index_" + ex_per + ".csv"
folder_name = './calibra/' + ex_per
# expression_list = ["none","happy","sad","fear","angry","surprise","disgusted","contempt"]
plot_title = ["accx","accy","accz","gyrox","gyroy","gyroz"]
expression_list = ["none","happy","sad","fear","angry","surprise","disgusted","contempt"]
def signalPlot(selected_data, expression, number):
    # 右侧 IMU 和参考 IMU 的数据
    right_imu_data = selected_data[:, 6:12]
    reference_imu_data = selected_data[:, 24:30]
    # 绘制图像，共 4x3 个子图
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # 设置标题
    fig.suptitle('Right IMU vs Reference IMU Signal('+ expression + '_' + str(number) + ')', fontsize=16)

    
    if not os.path.exists(folder_name):
        # 如果文件夹不存在，则创建它
        os.makedirs(folder_name)

    # 遍历每个子图并绘制相应的数据
    for i in range(2):
        for j in range(3):
            # 计算子图索引
            index = i*3 + j

            # 从右侧 IMU 数据中获取信号
            right_signal = right_imu_data[:, index]
            # 从参考 IMU 数据中获取信号
            reference_signal = reference_imu_data[:, index]
            # print(right_signal.shape,right_signal[4],type(right_signal[4]))
            # print(reference_signal.shape,reference_signal[4],type(reference_signal[4]))
            
            # 绘制右侧 IMU 信号
            axs[i, j].plot(right_signal, label='Right IMU')
            
            # 绘制参考 IMU 信号
            axs[i, j].plot(reference_signal, label='Reference IMU')
            # 设置图例
            axs[i, j].legend()
            # 设置子图标题
            axs[i, j].set_title(f'Signal of {plot_title[index]}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局
    # plt.show()  # 显示图像
    # 检查文件夹是否存在

    # 保存图像到文件
    file_name = f'Right_IMU({expression}_{number}).png'
    fig.savefig(os.path.join(folder_name, file_name))
    
    plt.cla()
    plt.close("all")

if __name__ == "__main__":
    epoch = 3
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
    imu_data[:,0:6] = unit_conversion(imu_data[:,0:6])
    imu_data[:,6:12] = unit_conversion(imu_data[:,6:12])
    imu_data[:,12:18] = unit_conversion(imu_data[:,12:18])
    

    imu_data[:,:] = remove_imu_peak(imu_data)
    
    # 映射到y为重力加速度为9.8的方向；
    cali__data = calibration(imu_data, index_data, time_data, imu_time, epoch)

    if cali__data is not None:
        proced_data = np.hstack((imu_data[:,:],cali__data))
    # imu_data[:,:] = mappingUniform(imu_data,index_data,time_data,imu_time)

    # segment
    for ex in range(len(index_data)):
        # expression_list[index_data[ex]] : category
        # 用于存储每个 epoch 计算结果的临时列表
        temp_correlations = []

        for ep in range(epoch):
            
            start_time = time_data[ex,ep*2]
            end_time = time_data[ex,ep*2+1]
            start_index = find_index(start_time,imu_time) - 200
            end_index = find_index(end_time,imu_time)
            # 片段低通滤波
            filtered_signal = filterData(proced_data[start_index-200:end_index,:])
            # 绘制图像
            signalPlot(filtered_signal,expression_list[index_data[ex]],ep)