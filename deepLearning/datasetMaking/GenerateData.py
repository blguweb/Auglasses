
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import os
from utility import *
import sys
import cv2
"""
@Description :  将采集的数据整理成imu和aul两个对齐csv文件
@Author      :  kidominox 
@Time        :   2024/01/10 14:48:14
"""

# 每一个人的数据先保存;切片；csv文件；

# 然后循环切片进行输入；

class GenerateData():
    def __init__(self, ex_per, pre_ex_per):
        self.ex_per = ex_per
        self.pre_ex_per = pre_ex_per

    def loadImuData(self):
        self.root_path = "./dataCollection/"
        imu_path = self.root_path +  "imu_" + self.ex_per + ".csv" 
        time_path = self.root_path + "time_" + self.ex_per + ".csv"
        index_path = self.root_path + "index_" + self.ex_per + ".csv"

        imu_raw_data = pd.read_csv(imu_path,header= None)
        imu_raw_data = imu_raw_data.values
        
        self.time_data = pd.read_csv(time_path,header= None)
        self.time_data = self.time_data.values

        # index: ["静止","开心","难过","害怕","生气","惊喜","厌恶","轻蔑"]的索引
        self.index_data = pd.read_csv(index_path,header= None)
        self.index_data = self.index_data.values
        self.index_data = self.index_data.flatten()

        self.imu_time = imu_raw_data[:,0]
        self.imu_data = imu_raw_data[:,2:]
        # time_data = time_data[:,:]
        self.imu_data[:,0:6] = unit_conversion(self.imu_data[:,0:6])
        self.imu_data[:,6:12] = unit_conversion(self.imu_data[:,6:12])
        self.imu_data[:,12:18] = unit_conversion(self.imu_data[:,12:18])

    def loadAUData(self):
        au_path = self.root_path + "video/"  + self.ex_per + ".csv"
        # calibration openface
        self.au_data = pd.read_csv(au_path)
        # 选出必要的进行合并
        self.au_data = self.au_data[[' timestamp',' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',\
            ' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r']]
        self.au_data = self.au_data.values



    def getFirstFrame(self):
        video_path = self.root_path + "video/"  + self.ex_per + ".mp4"
        self.is_clicked = False
        self.first_frame = self.video_capture(video_path = video_path)

    
    def calculate_maximum(self, epoch = 3):
        calibra_imu_path = self.root_path + "imu_" + self.pre_ex_per + ".csv"
        calibra_time_path = self.root_path + "time_" + self.pre_ex_per + ".csv"
        calibra_index_path = self.root_path + "index_" + self.pre_ex_per + ".csv"
        calibra_video_path = self.root_path + "video/"  + self.pre_ex_per + ".mp4"
        calibra_openface_path = self.root_path + "video/"  + self.pre_ex_per + ".csv"

        calibra_imu_data = pd.read_csv(calibra_imu_path,header= None)
        calibra_imu_data = calibra_imu_data.values

        calibra_time_data = pd.read_csv(calibra_time_path,header= None)
        calibra_time_data = calibra_time_data.values

        calibra_index_data = pd.read_csv(calibra_index_path,header= None)
        calibra_index_data = calibra_index_data.values
        calibra_index_data = calibra_index_data.flatten()

        calibra_imu_time = calibra_imu_data[:,0]
        calibra_imu_data = calibra_imu_data[:,2:]
        calibra_time_data = calibra_time_data[:,:]
        calibra_imu_data[:,0:6] = unit_conversion(calibra_imu_data[:,0:6])
        calibra_imu_data[:,6:12] = unit_conversion(calibra_imu_data[:,6:12])
        calibra_imu_data[:,12:18] = unit_conversion(calibra_imu_data[:,12:18])

        calibra_imu_data[:,:] = remove_imu_peak(calibra_imu_data)
        calibra_imu_data[:,:], self.R_left, self.T_left, self.R_right, self.T_right = mappingUniform(calibra_imu_data,calibra_index_data,calibra_time_data,calibra_imu_time)
        calibra_imu_data = filterData(calibra_imu_data)
        
        # calibration openface
        calibra_openface_data = pd.read_csv(calibra_openface_path)
        # 选出必要的进行合并
        calibra_openface_data = calibra_openface_data[[' timestamp',' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',\
            ' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r']]
        calibra_openface_data = calibra_openface_data.values
        self.is_clicked = False
        first_frame = self.video_capture(video_path = calibra_video_path)

        
        for ex in range(len(calibra_index_data)):
            # expression_list[index_data[ex]] : category
            # 用于存储每个 epoch 计算结果的临时列表
            epoch_maximum_acc = []
            epoch_maximum_gyro = []
            if ex == 3: # open mouth
                for ep in range(epoch):
                    # identified imu part
                    start_time = calibra_time_data[ex,ep*2]
                    end_time = calibra_time_data[ex,ep*2+1]
                    start_index = find_index(start_time,calibra_imu_time)
                    end_index = find_index(end_time,calibra_imu_time)

                    # 计算对应的视频帧范围
                    start_frame, end_frame = calculate_frame_range_from_timestamps(start_time, end_time, calibra_imu_time[0], first_frame, calibra_openface_data)

                    # 提取最大值进行归一化
                    imu_acc_max = np.max(abs(calibra_imu_data[start_index:end_index,6:9]))
                    imu_gyro_max = np.max(abs(calibra_imu_data[start_index:end_index,9:12]))

                    imu_acc_max_left = np.max(abs(calibra_imu_data[start_index:end_index,0:3]))
                    imu_gyro_max_left = np.max(abs(calibra_imu_data[start_index:end_index,3:6]))
                    print("imu_acc_max_left",imu_acc_max_left,"imu_acc_max",imu_acc_max)
                    print("imu_gyro_max_left",imu_gyro_max_left,"imu_gyro_max",imu_gyro_max)

                    # au max
                    # 从所有曲线中提取最大值
                    au_max = np.max(calibra_openface_data[start_frame:end_frame, 1:])
                    print("au_max",au_max)
                    if au_max > 2.5:
                        epoch_maximum_acc.append(imu_acc_max / (au_max / 5))
                        epoch_maximum_acc.append(imu_acc_max_left / (au_max / 5))
                        epoch_maximum_gyro.append(imu_gyro_max / (au_max / 5))
                        epoch_maximum_gyro.append(imu_gyro_max_left / (au_max / 5))
        return np.mean(epoch_maximum_acc), np.mean(epoch_maximum_gyro)
    


    def preProcessImuData(self):
        # remove peak
        self.imu_data[:,:] = remove_imu_peak(self.imu_data)
        # 参考imu进行伪影去除
        left_calibration = np.dot(self.R_left, self.imu_data[:,12:18].T) + np.tile(self.T_left.reshape(-1, 1), (1, self.imu_data.shape[0]))
        right_calibration = np.dot(self.R_right, self.imu_data[:,12:18].T) + np.tile(self.T_right.reshape(-1, 1), (1, self.imu_data.shape[0]))
        left_calibration = np.transpose(left_calibration)
        right_calibration = np.transpose(right_calibration)
        self.imu_data[:,0:6] = self.imu_data[:,0:6] - left_calibration
        self.imu_data[:,6:12] = self.imu_data[:,6:12] - right_calibration
        # 低通滤波
        self.imu_data = filterData(self.imu_data)
        
        # print("calibra_maximum","iden_maximum",self.calibra_maximum,self.iden_maximum)
        # 归一化
        self.imu_data = normalization(self.imu_data, acc_max = self.acc_maximum, gyro_max = self.gyro_maximum)
        

        # # 除去9.8的分量
        # acc = static_xyz(self.calibra_imu_data, self.calibra_index_data, self.calibra_time_data, self.calibra_imu_time, self.epoch)
        # if acc is not None:
        #     self.calibra_imu_data[:, 6:9] -= acc
        # acc = static_xyz(self.imu_data, self.index_data, self.time_data, self.imu_time, self.epoch)
        # if acc is not None:
        #     self.imu_data[:, 6:9] -= acc

        # 归一化
        # self.calibra_maximum = calculate_maximum(self.calibra_imu_data, self.calibra_index_data, self.calibra_time_data, self.calibra_imu_time, self.calibra_first_frame, self.calibra_au_data, self.epoch)
        
    
    def uniformTime(self):
        imu_start_dt = datetime.strptime(self.imu_time[0], '%H:%M:%S.%f')
        for i in range(self.imu_time.shape[0]):
            imu_each = datetime.strptime(self.imu_time[i], '%H:%M:%S.%f')
            self.imu_time[i] = (imu_each - imu_start_dt).total_seconds()
        
        au_start_dt = self.au_data[self.first_frame,0]
        self.au_data[:,0] = self.au_data[:,0] - au_start_dt
   
    def csvWriter(self,data,path):
        with open(path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # 遍历矩阵的每一行，写入CSV文件
            for row in data:
                csv_writer.writerow(row)

    def alignImuAndAu(self):
        # 先把ex_per的全部数据提取出来
        start_time = self.time_data[0,0]
        end_time = self.time_data[-1,-1]
        start_index = find_index(start_time,self.imu_time)
        end_index = find_index(end_time,self.imu_time) + 800

        self.uniformTime()
        
        # save 
        print(start_index, end_index)
        print(self.imu_time.shape,self.imu_data.shape)
        self.imu_time = self.imu_time.reshape(-1, 1)
        imu_all_data = np.hstack((self.imu_time[start_index:end_index],self.imu_data[start_index:end_index,0:12]))
        self.csvWriter(imu_all_data,"./dataset/imu/imu_" + self.ex_per + ".csv")

        au_start_index = find_index(self.imu_time[start_index],self.au_data[:,0])
        au_end_index = find_index(self.imu_time[end_index],self.au_data[:,0])
        print("au_start_index","au_end_index",au_start_index,au_end_index)
        au_all_data = self.au_data[au_start_index:au_end_index,:]
        self.csvWriter(au_all_data,"./dataset/au/au_" + self.ex_per + ".csv")

    def mouse_click(self, event, x, y, flags, para):
        if event == cv2.EVENT_LBUTTONDOWN: # 左边鼠标点击
            self.is_clicked = True


    def video_capture(self, video_path):
        # cap = cv2.VideoCapture(0)  #读取摄像头
        print(video_path)
        cv2.namedWindow("Capture")
        cv2.setMouseCallback("Capture", self.mouse_click)
        # 注意颜色得值 low < upper
        cap = cv2.VideoCapture(video_path)  #读取视频文件
        # red color
        lower_red = np.array([168,47, 255])
        upper_red = np.array([174,48,255])

        # lower_red = np.array([176 ,221 ,255])
        # upper_red = np.array([19 ,95, 255])
        frame_id = 0
        frame_start = 0
        no_start = True
        during = 0
        while(True):
            ret, frame = cap.read()
            if ret:
                [height,width,pixels] = frame.shape
                # 局部看
                frame = frame[0:height,0:width]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_red, upper_red)
                # cv2.imshow('Mask', mask)
                res = cv2.bitwise_and(frame, frame, mask=mask)
                # cv2.rectangle(res, (0, height), (width , 0 ), (0, 255, 0), 5)
                cv2.imshow('Result', res)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
                mask = cv2.dilate(mask, kernel)
                cnts1,hierarchy1=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#轮廓检测
                index = 0
                # print(len(cnts1))
                for cnt in cnts1:
                    (x,y,w,h)=cv2.boundingRect(cnt)#该函数返回矩阵四个点
                    # if(w*h >= 10):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)#将检测到的颜色框起来
                    cv2.putText(frame,'red',(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    index += 1
                if index >= 1 and no_start:
                    # cv2.putText(frame,'start',(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                    
                    frame_start = frame_id
                    no_start = False
                if not no_start:
                    during += 1
                if during > 2000:
                    break
                if self.is_clicked:
                    frame_start = frame_id
                    break

                frame_id += 1

                cv2.imshow('Capture', frame)
                cv2.waitKey(20)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        print("frame start",frame_start)

        cap.release()
        cv2.destroyAllWindows()
        return frame_start
        
    def generateData(self):
        self.loadImuData()
        self.loadAUData()
        self.getFirstFrame()
        self.acc_maximum, self.gyro_maximum = self.calculate_maximum()
        self.preProcessImuData()
        self.alignImuAndAu()

if __name__ == "__main__":

    gd = GenerateData("au_sys1","pre_sys1")
    gd.generateData()