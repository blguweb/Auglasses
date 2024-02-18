import serial
import sys
import csv
import signal
import pyqtgraph as pg
import array
import serial
import threading
import numpy as np
from queue import Queue
from datetime import datetime

"""
@Description :  将imu信号实时呈现图形化,测试信号的同步性————目测的形式
@Author      :  kidominox 
@Time        :   2022/07/15 16:11:17
"""
"""
data:

    time 
    eog_vertical
    eog_horizon
    up  acc x y z
        gyro x y z 
    left acc x y z
            gyro x y z
    right acc x y z
            gyro x y z
"""

portx = 'COM6'
path = "./imu_del1" +".csv"
data_head = ["time","imu_left_accx","imu_left_accy","imu_left_accz","imu_left_gyrox","imu_left_gyroy","imu_left_gyroz",
    "imu_right_accx","imu_right_accy","imu_right_accz","imu_right_gyrox","imu_right_gyroy","imu_right_gyroz","imu_head_accx","imu_head_accy","imu_head_accz","imu_head_gyrox","imu_head_gyroy","imu_head_gyroz"]
bps = 921600
i = 0
q_mag_imu_head = Queue(maxsize=0)
q_mag_imu_left = Queue(maxsize=0)
q_mag_imu_right = Queue(maxsize=0)
# 一个eog 左边 右边 上面的imu
# curve_num = 0
pre_data = [0] * 20

def unit_conversion_acc(raw_data):
    return (raw_data * 9.8) / 16384


def unit_conversion_gyro(raw_data):
    return (raw_data * 2000) / 0x8000


def parse_data(data):
    # print(data)
    data_list = []
    dt = datetime.today()
    data_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    data_list.append((data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3])
    for j in range(4,40,2):
        imu_data = (data[j] << 8) | data[j+1]
        if data[j] > 0x7F:
            imu_data = imu_data - (1 << 16)
        data_list.append(imu_data)
    pre_data[:] = data_list

def Serial():
    global q_mag_imu_head
    global q_mag_imu_left
    global q_mag_imu_right
    data_rec = b''
    databag_interval = b'\xEE\xAA'
    while True:
        if mSerial.inWaiting:
            # 如果有数据在等待接收，则接收并打印出来
            data_ = mSerial.read(mSerial.in_waiting)
            data_rec += data_
            data_split = data_rec.split(databag_interval)
            # print(len(data_split))
            if len(data_split) > 1:
                for i in range(len(data_split)-1):
                    if len(data_split[i]) == 40:
                        parse_data(data_split[i])
                    # get data
                    q_mag_imu_left.put(float(unit_conversion_acc(pre_data[8])))
                    q_mag_imu_right.put(float(unit_conversion_acc(pre_data[9]))-3)
                    q_mag_imu_head.put(float(unit_conversion_acc(pre_data[10])))
                    # print("left: ",unit_conversion_acc(pre_data[8]),unit_conversion_acc(pre_data[9]),unit_conversion_acc(pre_data[10]))
            data_rec = data_split[-1]


                # 测试一轴的同步性就行

                # pattern = re.compile(r"[+-]?\d+(?:\.\d+)?")   # find the num
                # data_all = pattern.findall(data_get)
                # print(data_all)

                # # save data
                # data_real = []
                # # time
                # dt = datetime.today()
                # data_real.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
                # # left imu
                # [data_real.append(float(data_split[j])) for j in range(1,19)]
                # print("data save: ", data_real)
                # write_csv(data_real)


def plotData():
    global i
    if i < historyLength:
        imu_head[i] = q_mag_imu_head.get()
        imu_left[i] = q_mag_imu_left.get()
        imu_right[i] = q_mag_imu_right.get()
        i = i+1
    else:
        imu_head[:-1] = imu_head[1:]
        imu_head[i-1] = q_mag_imu_head.get()
        
        imu_left[:-1] = imu_left[1:]
        imu_left[i-1] = q_mag_imu_left.get()

        imu_right[:-1] = imu_right[1:]
        imu_right[i-1] = q_mag_imu_right.get()

    curve4.setData(imu_head)
    curve5.setData(imu_left)
    curve6.setData(imu_right)

def write_csv(data_row):
                           
    with open(path,mode='a',newline = '',encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def sig_handler(signum, frame):
    sys.exit(0)

if __name__ == "__main__":
    # curve_num = 6 # 线的个数
    
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    app = pg.mkQApp()           # App Setup
    win = pg.GraphicsWindow()   # Window Setup
    win.setWindowTitle(u'pyqtgraph chart tool')
    win.resize(900, 600)        #window size

    imu_head = array.array('f')
    imu_left = array.array('f')
    imu_right  = array.array('f')
    historyLength = 1000 # 可看的长度

    imu_head = np.zeros(historyLength).__array__('f')
    imu_left = np.zeros(historyLength).__array__('f')
    imu_right = np.zeros(historyLength).__array__('f')

    write_csv(data_head)


    p2 = win.addPlot(title='imu')
    p2.showGrid(x=True, y=True)
    p2.setRange(xRange=[0, historyLength], yRange=[0,6], padding=0)
    p2.setLabel(axis='left',     text='y-mag')
    p2.setLabel(axis='bottom',   text='x-time')
    curve4 = p2.plot(imu_head, pen = 'g') # z
    curve5 = p2.plot(imu_left, pen = 'b') # x
    curve6 = p2.plot(imu_right, pen = 'm') # 紫红 y
    

    mSerial = serial.Serial(portx, int(bps))
    if (mSerial.isOpen()):
        print("open success")
        mSerial.flushInput()
    else:
        print("open failed")
        serial.close()
    
    #Serial data receive thread
    th1 = threading.Thread(target=Serial)
    th1.setDaemon(True)
    th1.start()
    
    #plot timer define
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(plotData)
    timer.start(10)
    app.exec_()

