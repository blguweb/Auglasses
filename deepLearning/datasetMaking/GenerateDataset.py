import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import os
from PIL import Image
from scipy import signal
"""
@Description :   数据集的制作 len -> 3秒； imu补充或者去掉 len x vchannel x 3s
@Author      :  kidominox 
@Time        :   2024/01/10 23:33:29
"""
# for PatchTST
class GenerateDataset(Dataset):
    def __init__(self, root_path = './dataset/', window_length:int = 46,
                 step_length:int = 23):
        
        self.window_length = window_length
        self.step_length = step_length
        self.root_path = root_path
        self.generateDataset()

    def find_index(self,target,imu_time):
        index = 0
        while index <= len(imu_time):
            if imu_time[index] >= target:
                break
            index += 1
        return index
    # 切割数据
    def slidingWindow(self,imu_data, au_data):

        X = []
        Y = []

        # 从AU数据开始滑动窗口
        for start_index in range(0, len(au_data), self.step_length):

            end_index = start_index + self.window_length
            if end_index > len(au_data):
                break

            # 获取当前窗口的AU数据
            current_au_window = au_data[start_index:end_index]

            # 获取对应的IMU数据段
            current_imu_segment = imu_data[self.find_index(current_au_window[0,0],imu_data[:,0]):self.find_index(current_au_window[-1,0],imu_data[:,0]),:]
            # current_imu_segment = imu_data[(imu_data[:,0] >= current_au_window[0,0]) & (imu_data[:,0] <= current_au_window[-1,0])]
            # print("ss",current_imu_segment.shape)

            # 检查IMU数据长度，执行删除或插值
            desired_length = 1200
            if current_imu_segment.shape[0] < desired_length:

                time_stamps = imu_data[:, 0]
                new_time_stamps = np.linspace(time_stamps[0], time_stamps[-1], desired_length)

                new_imu_data = []
                for i in range(1, imu_data.shape[1]):
                    # 对每列数据进行插值
                    interpolator = interp1d(time_stamps, imu_data[:, i], kind='linear')
                    new_column = interpolator(new_time_stamps)
                    new_imu_data.append(new_column)
            elif current_imu_segment.shape[0] > desired_length:
                # 裁剪数据,取中间的部分
                start_index = int((current_imu_segment.shape[0] - desired_length) / 2)
                end_index = start_index + desired_length
                new_imu_data = current_imu_segment[start_index:end_index, 1:]
            else:
                new_imu_data = current_imu_segment[:, 1:]
            
            new_au_data = current_au_window[:, 1:]

            X.append(new_imu_data)
            Y.append(new_au_data)
        return X, Y
        
    # 整合成一个数据集
    def generateDataset(self):
        # 所有的数据集
        self.dataset_x = []
        self.dataset_y = []
        imu_path = os.path.join(self.root_path, "imu")
        for file_name in os.listdir(imu_path):
            file_name_parts = file_name.split('_')
            if len(file_name_parts) >= 3:
                extracted_text = "au_" + file_name_parts[1] + '_' + file_name_parts[2]
                # print("extracted_text",extracted_text)
                
            imu_data = pd.read_csv(os.path.join(imu_path, file_name), header=None).values
            au_data = pd.read_csv(os.path.join(self.root_path, "au", extracted_text), header=None).values
            # 切割数据
            personal_x, personal_y = self.slidingWindow(imu_data, au_data)
            print(len(personal_x), len(personal_y))
            # 添加到数据集
            if self.dataset_x == []:
                self.dataset_x = personal_x
                self.dataset_y = personal_y
            else:
                self.dataset_x = np.vstack((self.dataset_x, personal_x))
                self.dataset_y = np.vstack((self.dataset_y, personal_y))
        print("dataset_x shape:", len(self.dataset_x))



    def readData(self, flag = 'train'):
        # 根据类别分割数据集
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[flag]
        dataset_length = len(self.dataset_x)
        num_train = int(dataset_length * 0.7)
        num_test = int(dataset_length * 0.2)
        num_vali = dataset_length - num_train - num_test

        border1s = [0, num_train, dataset_length - num_test]
        border2s = [num_train, num_train + num_vali, dataset_length]
        border1 = border1s[set_type]
        border2 = border2s[set_type]
        self.data_x = self.dataset_x[border1:border2]
        self.data_y = self.dataset_y[border1:border2]

    def __getitem__(self, index):

        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)



# for coatnet
class My_Dataset(Dataset):
    """自定义数据集"""

    def __init__(self, transform=None, window_length=800, root_path='./dataset/'):
        self.transform = transform
        self.window_length = window_length
        self.root_path = root_path
        self.generateDataset()


    def interpolate_imu(self, imu_data, target_length=4000):
        time_stamps = imu_data[:, 0]
        new_time_stamps = np.linspace(time_stamps[0], time_stamps[-1], target_length)

        new_imu_data = []
        for i in range(1, imu_data.shape[1]):
            # 对每列数据进行插值
            interpolator = interp1d(time_stamps, imu_data[:, i], kind='linear')
            new_column = interpolator(new_time_stamps)
            new_imu_data.append(new_column)
        new_imu_data = np.transpose(np.array(new_imu_data))
        return new_imu_data

    # 切割数据
    def slidingWindow(self,imu_data, au_data, window_length=800):

        segments = []

        for index in range(au_data.shape[0]):
            start_index = index - window_length//2
            end_index = start_index + window_length

            # Selecting the imu data in the window
            window_imu_data = imu_data[self.find_index(au_data[start_index,0],imu_data[:,0]),
                                       :self.find_index(au_data[end_index,0],imu_data[:,0]),:]

            if len(window_imu_data) < 2:  # Check to ensure there are at least two points for interpolation
                continue

            # Interpolating the imu data to a fixed length of 4000
            interpolated_imu_data = self.interpolate_imu(window_imu_data, target_length=4000)

            # spectrogram
            images = self.get_spectrogram(interpolated_imu_data)
            # Adding the pair (X, y) to the dataset
            segments.append((images, np.array(au_data[1:])))  # Assuming the rest of the columns are features

        return segments


    # 整合成一个数据集
    def generateDataset(self):
        # 如果不存在数据集文件，则生成数据集
        if os.path.exists("./dataset.npy"):
            self.dataset = np.load("./dataset.npy", allow_pickle=True)
            self.dataset = self.dataset.item()
            print("load dataset",type(self.dataset),len(self.dataset))
            return
        # 所有的数据集
        self.dataset = []
        imu_path = os.path.join(self.root_path, "imu")
        for file_name in os.listdir(imu_path):
            file_name_parts = file_name.split('_')
            if len(file_name_parts) >= 3:
                extracted_text = "au_" + file_name_parts[1] + '_' + file_name_parts[2]
                # print("extracted_text",extracted_text)
                
            imu_data = pd.read_csv(os.path.join(imu_path, file_name), header=None).values
            au_data = pd.read_csv(os.path.join(self.root_path, "au", extracted_text), header=None).values
            # 切割数据
            single_dataset = self.slidingWindow(imu_data, au_data,window_length=self.window_length)
            print("single dataset", len(single_dataset))
            # 添加到数据集
            self.dataset.extend(single_dataset)
        print("dataset shape:", len(self.dataset))
        # save dataset
        np.save("./dataset.npy", self.dataset)


    # 求出数据段的时频图作为输入
    def get_spectrogram(self, signal_array, sampling_rate=400, padding=4000, noverlap=32, nperseg=64):
        
        channels_spect = []
        for i in range(signal_array.shape[1]):  # 遍历前12列
            signal_data = signal_array[:,i]
            print("signal_data.shape:",signal_data.shape)
            # Compute the spectrogram
            frequencies, times, spectrogram = signal.spectrogram(signal_data, fs=sampling_rate, nfft=padding,
                                                                  noverlap=noverlap, nperseg=nperseg, mode='magnitude')
            print(spectrogram.shape)

            # Normalize the spectrogram
            spectrogram_normalized = np.log(spectrogram.astype('float') + 1e-7)  # Adding a small constant to avoid log(0)

            # Find indices corresponding to the frequency range 0-20 Hz
            # freq_range = (frequencies >= 0) & (frequencies <= 20)
            # spectrogram_extracted = spectrogram_normalized[freq_range, :]

            # Resize the extracted spectrogram to a square shape and duplicate it across 3 channels
            x_size = 224  # Example size, you can adjust this
            spectrogram_image = Image.fromarray(spectrogram_normalized)
            spectrogram_resized = spectrogram_image.resize((x_size, x_size))
            spectrogram_resized_array = np.array(spectrogram_resized)
            # spectrogram_3d = np.repeat(spectrogram_resized_array[..., np.newaxis], 3, axis=2)

            channels_spect.append(spectrogram_resized_array)
            # Visualizing the resized extracted spectrogram
            plt.figure(figsize=(6, 6))
            plt.imshow(spectrogram_resized_array, aspect='auto', cmap='hot')
            plt.title('Resized Extracted Spectrogram (0-20 Hz)')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar(label='Intensity')
            plt.show()
        channels_spect = np.array(channels_spect)
        return channels_spect
    
    # 分割数据集
    def readData(self, flag = 'train'):
        # 根据类别分割数据集
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[flag]
        dataset_length = len(self.dataset_x)
        num_train = int(dataset_length * 0.7)
        num_test = int(dataset_length * 0.2)
        num_vali = dataset_length - num_train - num_test

        border1s = [0, num_train, dataset_length - num_test]
        border2s = [num_train, num_train + num_vali, dataset_length]
        border1 = border1s[set_type]
        border2 = border2s[set_type]
        self.data_x = self.dataset[border1:border2][0]
        self.data_y = self.dataset[border1:border2][1]


    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.images_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":
    dataset = GenerateDataset(root_path='./dataset/', window_length=46, step_length=5)
    # dataset.readData(flag='train')
