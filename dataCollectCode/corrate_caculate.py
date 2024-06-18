import numpy as np
import pandas as pd
from utility import *
import sys
import cv2
"""
@Description :  判断左右信号是否符合要求
@Author      :  kidominox 
@Time        :   2024/01/04 11:39:00
"""


class SignalJudge:
    def __init__(self, ex_per):
        # self.calibra_ex_per = 's'
        self.epoch = 3
        # self.calibra_imu_path = "./data/imu_" + self.calibra_ex_per + ".csv" 
        # self.calibra_time_path = "./data/time_" + self.calibra_ex_per + ".csv"
        # self.calibra_index_path = "./data/index_" + self.calibra_ex_per + ".csv"
        # self.calibra_au_path = "./video/" + self.calibra_ex_per + ".csv"
        # self.calibra_video_path = "./video/" + self.calibra_ex_per + ".avi"

        self.imu_path = "./dataCollection/imu_" + ex_per + ".csv" 
        self.time_path = "./dataCollection/time_" + ex_per + ".csv"
        self.index_path = "./dataCollection/index_" + ex_per + ".csv"
        # self.iden_au_path = "./video/" + ex_per + ".csv"
        # self.iden_video_path = "./video/" + ex_per + ".avi"

        self.expression_list = ["none","happy","frown","open mouth"]


    def calibraData(self):
        # 读取npy文件
        self.calibra_signal = np.load("calibra_signal.npy", allow_pickle=True)
        # 将加载的数据转换为字典
        self.calibra_signal = self.calibra_signal.item()



    def loadImuData(self):
        # calibra_imu_data = pd.read_csv(self.calibra_imu_path,header= None)
        # calibra_imu_data = self.calibra_imu_data.values
        # # [start time, end time] * epoch
        # self.calibra_time_data = pd.read_csv(self.calibra_time_path,header= None)
        # self.calibra_time_data = self.calibra_time_data.values
        # # index: 索引
        # self.calibra_index_data = pd.read_csv(self.calibra_index_path,header= None)
        # self.calibra_index_data = self.calibra_index_data.values
        # self.calibra_index_data = self.calibra_index_data.flatten()

        # self.calibra_imu_time = calibra_imu_data[:,0]
        # self.calibra_imu_data = calibra_imu_data[:,2:]
        # # calibra_time_data = calibra_time_data[:,:]
        # self.calibra_imu_data[:,0:6] = unit_conversion(self.calibra_imu_data[:,0:6])
        # self.calibra_imu_data[:,6:12] = unit_conversion(self.calibra_imu_data[:,6:12])
        # self.calibra_imu_data[:,12:18] = unit_conversion(self.calibra_imu_data[:,12:18])


        # identified imu part
        imu_raw_data = pd.read_csv(self.imu_path,header= None)
        imu_raw_data = imu_raw_data.values[1:,:]
        # [start time, end time] * epoch
        self.time_data = pd.read_csv(self.time_path,header= None)
        self.time_data = self.time_data.values
        # index: ["静止","开心","难过","害怕","生气","惊喜","厌恶","轻蔑"]的索引
        self.index_data = pd.read_csv(self.index_path,header= None)
        self.index_data = self.index_data.values
        self.index_data = self.index_data.flatten()

        self.imu_time = imu_raw_data[:,0]
        self.imu_data = imu_raw_data[:,2:]
        # time_data = time_data[:,:]
        self.imu_data[:,0:6] = unit_conversion(self.imu_data[:,0:6])
        self.imu_data[:,6:12] = unit_conversion(self.imu_data[:,6:12])
        self.imu_data[:,12:18] = unit_conversion(self.imu_data[:,12:18])

    def loadAUData(self):
        # calibration openface
        self.calibra_au_data = pd.read_csv(self.calibra_au_path)
        # 选出必要的进行合并
        self.calibra_au_data = self.calibra_au_data[[' timestamp',' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',\
            ' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r']]
        self.calibra_au_data = self.calibra_au_data.values
        self.is_clicked = False
        self.calibra_first_frame = self.video_capture(video_path = self.calibra_video_path)
        # calibra_first_frame = 384

        # identify data openface and video
        self.iden_au_data = pd.read_csv(self.iden_au_path)
        # 选出必要的进行合并
        self.iden_au_data = self.iden_au_data[[' timestamp',' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',\
            ' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r']]
        self.iden_au_data = self.iden_au_data.values
        self.is_clicked = False
        self.iden_first_frame = self.video_capture(video_path = self.iden_video_path)

    def preProcessImuData(self):
        # remove peak
        # self.calibra_imu_data[:,:] = remove_imu_peak(self.calibra_imu_data)
        self.imu_data[:,:] = remove_imu_peak(self.imu_data)
        # 参考imu进行伪影去除
        self.imu_data,_,_,_,_ = mappingUniform(self.imu_data,self.index_data,self.time_data,self.imu_time)
        # self.calibra_imu_data = mappingUniform(self.calibra_imu_data,self.calibra_index_data,self.calibra_time_data,self.calibra_imu_time)
        
        # # 除去9.8的分量
        # acc = static_xyz(self.calibra_imu_data, self.calibra_index_data, self.calibra_time_data, self.calibra_imu_time, self.epoch)
        # if acc is not None:
        #     self.calibra_imu_data[:, 6:9] -= acc
        # acc = static_xyz(self.imu_data, self.index_data, self.time_data, self.imu_time, self.epoch)
        # if acc is not None:
        #     self.imu_data[:, 6:9] -= acc

        # 归一化
        # self.calibra_maximum = calculate_maximum(self.calibra_imu_data, self.calibra_index_data, self.calibra_time_data, self.calibra_imu_time, self.calibra_first_frame, self.calibra_au_data, self.epoch)
        # self.iden_maximum = calculate_maximum(self.imu_data, self.index_data, self.time_data, self.imu_time, self.iden_first_frame, self.iden_au_data, self.epoch)
        # print("calibra_maximum","iden_maximum",self.calibra_maximum,self.iden_maximum)
    

    

    def judgeSignal(self):
        for ex in range(len(self.index_data)):
            # expression_list[index_data[ex]] : category
            # 用于存储每个 epoch 计算结果的临时列表
            left_correlations = []
            right_correlations = []
            if ex in [0]:
                continue
            for ep in range(self.epoch):
                print(self.expression_list[self.index_data[ex]],ep)
                # identified imu part
                start_time = self.time_data[ex,ep*2]
                end_time = self.time_data[ex,ep*2+1]
                start_index = find_index(start_time,self.imu_time) - 200
                end_index = find_index(end_time,self.imu_time) + 800
                # 片段低通滤波
                iden_filtered_signal = filterData(self.imu_data[start_index:end_index,:], 8)

                # 归一化
                # iden_filtered_signal = normalization(iden_filtered_signal,acc_max = self.iden_maximum)

                # 提取信号强度超过一定阈值时长的片段
                # calibra_during_signal = calculate_expression_duration(filtered_signal,threshold=4) / 0.05

                # st,end,calibra_during_signal = calculate_expression_duration(filtered_signal,threshold=0.05)
                # plot_signal_with_threshold(filtered_signal, (st,end), 0.05, expression_list[index_data[ex]])

                # calibration imu part
                # start_time = self.calibra_time_data[ex,ep*2]
                # end_time = self.calibra_time_data[ex,ep*2+1]
                # start_index = find_index(start_time,self.calibra_imu_time) - 200
                # end_index = find_index(end_time,self.calibra_imu_time)
                # # 片段低通滤波
                # calibra_filtered_signal = filterData(self.calibra_imu_data[start_index:end_index,:],8)
                # # 归一化
                # calibra_filtered_signal = normalization(calibra_filtered_signal,acc_max = self.calibra_maximum)

                # st,end,calibra_during_signal = calculate_expression_duration(calibra_filtered_signal, threshold=0.05)
                # plot_signal_with_threshold(filtered_signal, (st,end),2, expression_list[index_data[ex]])
                #插值
                # ca_interp, iden_interp = interpolation(calibra_during_signal,imu_during_signal)
                epoch_left_correlations = correlationCoefficient(self.calibra_signal[self.expression_list[self.index_data[ex]]], iden_filtered_signal, "left")
                epoch_right_correlations = correlationCoefficient(self.calibra_signal[self.expression_list[self.index_data[ex]]], iden_filtered_signal, "right")

                left_correlations.append(epoch_left_correlations)                                                                                                                     
                right_correlations.append(epoch_right_correlations)
            avg_left_correlations = np.mean(left_correlations, axis=0)
            avg_right_correlations = np.mean(right_correlations, axis=0)
            correlationsJudge(avg_left_correlations, avg_right_correlations, self.expression_list[self.index_data[ex]])



    def mouse_click(self, event, x, y, flags, para):
        if event == cv2.EVENT_LBUTTONDOWN: # 左边鼠标点击
            self.is_clicked = True


    def video_capture(self, video_path):
        # cap = cv2.VideoCapture(0)  #读取摄像头\
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
                if during > 300:
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
        
    def processJudgement(self):
        self.loadImuData()
        self.calibraData()
        # self.loadAUData()
        self.preProcessImuData()
        self.judgeSignal()
        

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ex_per = sys.argv[1]
    # ex_per = "s"
    signal_judge = SignalJudge(ex_per)
    signal_judge.processJudgement()
    # 下载npy文件

