import csv
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, lfilter
import matplotlib.pyplot as plt
imu_path = "./dataCollection/imu_ex_yh1.csv" 
time_path = ".//dataCollection/time_ex_yh1.csv" 
picture_path = "./presee_signal/"
feature = ["imu_left_accx","imu_left_accy","imu_left_accz","imu_left_gyrox","imu_left_gyroy","imu_left_gyroz",
"imu_right_accx","imu_right_accy","imu_right_accz","imu_right_gyrox","imu_right_gyroy","imu_right_gyroz","imu_head_accx","imu_head_accy","imu_head_accz","imu_head_gyrox","imu_head_gyroy","imu_head_gyroz"]
ex = "yx1"
head = ["happy","sad","fear","angry","surprise","disgusted","none","time"]
epoch = 3
window = 450

def unit_conversion(raw_data):
    raw_data[:,0] = (raw_data[:,0] * 9.8) / 16384
    raw_data[:,1] = (raw_data[:,1] * 9.8) / 16384
    raw_data[:,2] = (raw_data[:,2] * 9.8) / 16384
    raw_data[:,3] = (raw_data[:,3] * 2000) / 0x8000
    raw_data[:,4] = (raw_data[:,4] * 2000) / 0x8000
    raw_data[:,5] = (raw_data[:,5] * 2000) / 0x8000
    return raw_data


def plot_signals(imu_data,imu_time,time_data):

    for j in range(len(head)):
        fig=plt.figure(figsize=(10, 15))
        imu_left_ax=fig.add_subplot(6,1,1)
        imu_left_g_ax = fig.add_subplot(6,1,2)
        imu_right_ax=fig.add_subplot(6,1,3)
        imu_right_g_ax = fig.add_subplot(6,1,4)
        imu_head_ax=fig.add_subplot(6,1,5)
        imu_head_g_ax=fig.add_subplot(6,1,6)
        imu_t = 0
        while imu_time[imu_t] < time_data[j][1]:
            imu_t += 1
        start_time = imu_t
        while imu_time[imu_t] < time_data[j][2]:
            imu_t += 1
        end_time = imu_t
        x = np.arange(0,end_time - start_time)
        print(start_time,end_time)
        
        if time_data[j][0] == -1:
            category = len(head)-1
        else:
            category = time_data[j][0]

        for k in range(0,3):
            imu_left_ax.plot(x,imu_data[start_time:end_time,k],label = feature[k])
        imu_left_ax.legend(bbox_to_anchor=(1,1), loc="upper left")
        # imu_left_ax.set_ylim([-10, 10])
        imu_left_ax.set_title(head[category] +'_'+"imu")

        for k in range(3,6):
            imu_left_g_ax.plot(x,imu_data[start_time:end_time,k],label = feature[k])
        imu_left_g_ax.legend(bbox_to_anchor=(1,1), loc="upper left")
        # imu_left_g_ax.set_ylim([-10, 10])
        imu_left_g_ax.set_title(head[category] +'_' +"imu")
        
        for k in range(6,9):
            imu_right_ax.plot(x,imu_data[start_time:end_time,k],label = feature[k])
        imu_right_ax.legend(bbox_to_anchor=(1,1), loc="upper left")
        # imu_right_ax.set_ylim([-10, 10])
        imu_right_ax.set_title(head[category] +'_' +"imu")

        for k in range(9,12):
            imu_right_g_ax.plot(x,imu_data[start_time:end_time,k],label = feature[k])
        imu_right_g_ax.legend(bbox_to_anchor=(1,1), loc="upper left")
        # imu_right_g_ax.set_ylim([-10, 10])
        imu_right_g_ax.set_title(head[category] + '_' +"imu")

        for k in range(12,15):
            imu_head_ax.plot(x,imu_data[start_time:end_time,k],label = feature[k])
        imu_head_ax.legend(bbox_to_anchor=(1,1), loc="upper left")
        # imu_head_ax.set_ylim([-10, 10])
        imu_head_ax.set_title(head[category] + '_' +"imu")
        
        for k in range(15,18):
            imu_head_g_ax.plot(x,imu_data[start_time:end_time,k],label = feature[k])
        imu_head_g_ax.legend(bbox_to_anchor=(1,1), loc="upper left")
        # imu_head_g_ax.set_ylim([-20, 20])
        imu_head_g_ax.set_title(head[category] + '_' +"imu")

        plt.tight_layout()
        # plt.savefig(picture_path + head[category] + '_'  + ex +".jpg")
        plt.show()

def mapping_method(A,B):
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

def mapping_uniform(data):
    left_data =data[start_filted:end_filted,0:6]
    right_data = data[start_filted:end_filted,6:12]
    calibration_data = data[start_filted:end_filted,12:18]
    left_data = np.float64(left_data)
    right_data = np.float64(right_data)
    calibration_data = np.float64(calibration_data)

    N = end_filted - start_filted
    print(type(left_data[0,0]))
    R_left,T_left = mapping_method(calibration_data,left_data)
    R_right,T_right = mapping_method(calibration_data,right_data)




    left_calibration = np.dot(R_left, data[:,12:18].T) + np.tile(T_left.reshape(-1, 1), (1, data.shape[0]))
    right_calibration = np.dot(R_right, data[:,12:18].T) + np.tile(T_right.reshape(-1, 1), (1, data.shape[0]))
    left_calibration = np.transpose(left_calibration)
    right_calibration = np.transpose(right_calibration)

    # left - calibration
    data[:,0:6] = data[:,0:6] - left_calibration
    data[:,6:12] = data[:,6:12] - right_calibration

    return data


def remove_imu_peak(imu_data):
    # data (length,imu category)
    for i in range(imu_data.shape[1]):
        if i in [0,1,2,6,7,8,12,13,14]:
            theta = 4
        else:
            theta =12
        index = ((imu_data[:,i] - np.roll(imu_data[:,i],1) > theta) & (imu_data[:,i] - np.roll(imu_data[:,i],-1) > theta)) | ((imu_data[:,i] - np.roll(imu_data[:,i],1) < -theta) & (imu_data[:,i] - np.roll(imu_data[:,i],-1) < -theta))
        plt.plot(imu_data[:,i])
        plt.plot(index.nonzero()[0], imu_data[index,i], 'ro')
        plt.show()
        index = np.nonzero(index)
        print(index)
        for j in index:
            print(j)
            imu_data[j,i] = (imu_data[j+1,i] +imu_data[j-1,i]) / 2
    return imu_data

# Function for low-pass filtering
def low_pass_filter(data, cutoff=20, fs=400, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


if __name__ == "__main__":

    imu_data = pd.read_csv(imu_path,header= None)
    imu_data = imu_data.values
    time_data = pd.read_csv(time_path,header= None)
    time_data = time_data.values
    imu_time = imu_data[:,0]
    imu_data = imu_data[:,2:]
    time_data = time_data[:,:]
    imu_data[:,0:6] = unit_conversion(imu_data[:,0:6])
    imu_data[:,6:12] = unit_conversion(imu_data[:,6:12])
    imu_data[:,12:18] = unit_conversion(imu_data[:,12:18])
    
    start_filted = 2250
    end_filted = 2300
    imu_data = remove_imu_peak(imu_data)
    # segment_signal = low_pass_filter(imu_data[:,8])
    imu_data = mapping_uniform(imu_data)
    plot_signals(imu_data,imu_time,time_data)