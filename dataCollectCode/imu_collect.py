import serial
import csv
import chardet
from datetime import datetime
import sys
'''
+---------+-------------------------------------------------+
| Header  | Data (40 bytes if valid)                       |
+---------+-------------------------------------------------+
| 0xEEAA  | ...                                            |
+---------+-------------------------------------------------+

'''

class DataProcessor:

    def __init__(self, com_port='COM6', baud_rate=921600):
        self.ser = serial.Serial(com_port, baud_rate)
        self.csv_file = open(imu_path, 'a', newline='')
        self.writer = csv.writer(self.csv_file)
        self.start_feature = 0
        self.start_collect = False
        self.pre_data = [0] * 20

    def read_data(self):
        data_rec = b''
        databag_interval = b'\xEE\xAA'
        while True:
            if self.ser.in_waiting:
                # 如果有数据在等待接收，则接收并打印出来
                data_ = self.ser.read(self.ser.in_waiting)
                data_rec += data_
                data_split = data_rec.split(databag_interval)
                print(len(data_split))
                if len(data_split) > 1:
                    for i in range(len(data_split)-1):
                        if len(data_split[i]) == 40:
                            self.parse_data(data_split[i])
                        else:
                            if self.pre_data != [0] * 20:
                                self.write_csv(self.pre_data)
                data_rec = data_split[-1]
            
                    

    def parse_data(self, data):
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
        self.pre_data[:] = data_list
        self.write_csv(data_list)

    def write_csv(self, data_list):
        self.writer.writerow(data_list)

    def close(self):
        self.csv_file.close()
        self.ser.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ex_per = sys.argv[1]
        imu_path = "./dataCollection/imu_"+ ex_per +".csv"
    else:
        print("No command-line arguments provided.")
    dp = DataProcessor()
    try:
        dp.read_data()
    except KeyboardInterrupt:
        dp.close()
