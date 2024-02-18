import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, lfilter
import matplotlib.pyplot as plt
"""
@Description :  对信号进行峰值异常去除，低通滤波，求取SNR
@Author      :  kidominox 
@Time        :   2023/11/28 11:25:04
"""
ex_per = 'd_3_10_4'
imu_path = "./selectSize/imu_" + ex_per + ".csv" 
time_path = "./selectSize/time_" + ex_per + ".csv"
index_path = "./selectSize/index_" + ex_per + ".csv"
feature = ["imu_left_accx","imu_left_accy","imu_left_accz","imu_left_gyrox","imu_left_gyroy","imu_left_gyroz",\
    "imu_right_accx","imu_right_accy","imu_right_accz","imu_right_gyrox","imu_right_gyroy","imu_right_gyroz",\
    "imu_head_accx","imu_head_accy","imu_head_accz","imu_head_gyrox","imu_head_gyroy","imu_head_gyroz"]
expression_list = ["none","happy","sad","fear","angry","surprise","disgusted","contempt"]

def remove_imu_peak(imu_data):
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
    

def calculate_rms(data):
    """计算数据的均方根"""
    return np.sqrt(np.mean(np.square(data)))

def Power_calculate(time_data,imu_time,imu_data,epoch):
    Power_epochs = []
    for i in range(epoch):
        start_time = time_data[i*2]
        end_time = time_data[i*2+1]
        # print(start_time,end_time)
        start_index = find_index(start_time,imu_time)-200
        end_index = find_index(end_time,imu_time)

        axes_Power = []
        # segment imu_data[start_index:end_index,:]
        # 18 axes
        # 要去除伪影；要基线去除；
        for j in range(imu_data.shape[1]):
            # low pass filter
            segment_signal = low_pass_filter(imu_data[start_index:end_index,j])
            # low pass filter 0.5Hz as baseline
            # baseline = low_pass_filter(imu_data[start_index:end_index,j],cutoff=0.5)
            segment_signal = segment_signal - np.mean(segment_signal,axis=0)
            # RMS
            rms_value = calculate_rms(segment_signal)
            axes_Power.append(rms_value)
        Power_epochs.append(axes_Power)
    return Power_epochs

def SNR_calculate(Power_matrix,index_data,epoch):
    # Power_matrix (8,5,18)
    # index_data (8,1)
    # epoch 5
    # Adjust the processing to handle the index correctly
    processed_matrix_corrected = np.zeros_like(Power_matrix[1:, :, :])
    # 1. 计算baseline
    baseline = []
    for i in range(len(expression_list)):
        if index_data[i] == 0:
            baseline.append(Power_matrix[i])
            break
    baseline = np.array(baseline).squeeze(axis=0)
    # baseline = np.mean(baseline,axis=0)
    # 2. 计算SNR
    for i in range(len(expression_list)):
        if index_data[i] == 0:
            continue
        processed_matrix_corrected[index_data[i]-1] = Power_matrix[i] / baseline
    processed_matrix_corrected = np.array(processed_matrix_corrected)
    print(processed_matrix_corrected.shape)
    # 对epoch求平均
    processed_matrix_corrected = np.mean(processed_matrix_corrected, axis=1)
    print(processed_matrix_corrected.shape)

    rms_values = []

    # Compute the RMS in blocks of 3 along the third dimension
    for i in range(0, processed_matrix_corrected.shape[1], 3):
        rms_block = np.sqrt(np.mean(np.square(processed_matrix_corrected[:, i:i+3]), axis=1))
        rms_values.append(rms_block)
    result_matrix = np.array(rms_values).T
    # 3. 保存到npy文件
    # np.save("./snr_matrix.npy",processed_matrix_corrected)
    
    # 4. 保存到csv文件
    # columns = ['left_acc',"left_gyro",'right_acc',"right_gyro",'head_acc',"head_gyro"]
    columns = ['right_acc',"right_gyro",'head_acc',"head_gyro"]
    # for i in range(epoch):
    #     for j in range(2):
    #         for k in range(6):
    #             columns.append("epoch"+str(i+1)+"_"+feature[k+6*j])
    columns = np.array(columns)
    # 保存到csv文件
    snr_table = pd.DataFrame(result_matrix[:,2:],
                                   index=expression_list[1:], columns=columns)
    snr_table.to_csv("./SNR/" + ex_per + ".csv")

def static_xyz(index_data, time_data, imu_time, imu_data, epoch):
    for i in range(len(expression_list)):
        if index_data[i] == 0:
            acc_mean_m = []
            for j in range(epoch):
                start_time = time_data[i, j*2]
                end_time = time_data[i, j*2+1]
                # print(start_time,end_time)
                start_index = find_index(start_time,imu_time)-200
                end_index = find_index(end_time,imu_time)
                # each_mean = []
                # for k in [6,7,8]: # acc
                #     # low pass filter
                #     segment_signal = low_pass_filter(imu_data[start_index:end_index,k])
                #     each_mean.append(np.mean(segment_signal))
                #     # print(segment_signal.shape)
                #     # print("acc", np.mean(segment_signal), j)
                # acc_mean.append(each_mean)
                acc_mean_m.append([
                        np.mean(low_pass_filter(imu_data[start_index:end_index, k]))
                        for k in [6, 7, 8]
                    ])
            acc_mean = np.mean(acc_mean_m, axis=0)
            # 方差
            acc_var = np.var(acc_mean_m, axis=0)
            print("acc_mean", acc_mean)
            print("acc_var", acc_var)



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
    imu_data[:,0:6] = unit_conversion(imu_data[:,0:6])
    imu_data[:,6:12] = unit_conversion(imu_data[:,6:12])
    imu_data[:,12:18] = unit_conversion(imu_data[:,12:18])

    imu_data = remove_imu_peak(imu_data)
    # imu_data = mappingUniform(imu_data,index_data,time_data)
    
    epoch = 5
    Power_matrix = []
    # segment
    for i in range(len(expression_list)):
        # expression_list[index_data[i]] : category
        Power_matrix.append(Power_calculate(time_data[i,:],imu_time,imu_data,epoch))

    print(np.array(Power_matrix).shape)
    
    # 保存到npy文件
    # np.save("./Power_matrix.npy",np.array(Power_matrix))

    # SNR
    SNR_calculate(np.array(Power_matrix),index_data,epoch)

    # calibration acc9.8 from none expression 
    static_xyz(index_data, time_data, imu_time, imu_data, epoch)

    # table 
