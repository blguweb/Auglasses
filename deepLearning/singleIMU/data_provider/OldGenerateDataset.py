import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import pickle
import os
"""
@Description :   数据集的制作 len -> 3秒； imu补充或者去掉 len x vchannel x 3s
@Author      :  kidominox 
@Time        :   2024/01/10 23:33:29
"""
def find_index(target,imu_time):
    index = 0
    while index < len(imu_time):
        if imu_time[index] >= target:
            break
        index += 1
    if index == len(imu_time) and imu_time[index-1] < target:
        return None
    return index

# 切割数据
def slidingWindow(imu_path, au_path, window_length, step_length, desired_length=1200):
    imu_data = pd.read_csv(imu_path, header=None).values
    au_data = pd.read_csv(au_path, header=None).values

    imu_units = []
    au_lists = []

    for index in range(30,au_data.shape[0]-60):
        m_index = find_index(au_data[index,0],imu_data[:,0])
        if m_index == None:
            continue
        start_index = m_index - desired_length//2
        end_index = start_index + desired_length
        if end_index >= imu_data.shape[0] or start_index < 0:
            continue
        # Selecting the imu data in the window
        window_imu_data = imu_data[start_index:end_index, :]
        if len(window_imu_data) < 2:  # Check to ensure there are at least two points for interpolation
            continue

        # Adding the pair (X, y) to the dataset
        imu_units.append(np.array(window_imu_data[:,1:]))  # Assuming the rest of the columns are features
        au_lists.append(np.array(au_data[index,1:]))
    print(len(au_lists))
    X = []
    Y = []
    A = []
    # 从AU数据开始滑动窗口
    for slid_index in range(0, len(au_lists), step_length):

        end_index = slid_index + window_length
        if end_index > len(au_lists):
            break

        # 获取当前窗口的AU数据
        current_au_window = au_lists[slid_index:end_index]
        
        current_imu_units = imu_units[slid_index:end_index]


        X.append(np.array(current_imu_units))
        Y.append(np.array(current_au_window))
        # 没有用的au 只是为了不报错
        A.append(np.array(current_au_window))
    print(len(X),len(Y))
    return X, Y, A

def dict_data(path_root):
    # 初始化数据结构来存储文件信息
    data_files = {}
    # 遍历数据集目录
    for file_name in os.listdir(path_root):
        # 分解文件名以获取人名和类型
        parts = file_name.split('_')
        if len(parts) >= 3:
            # 提取人名和类型（ex或au）
            person_name = parts[2][:3]  # 假设人名总是在第三部分的前3个字符
            file_type = '_'.join(parts[:2])  # 文件类型组合（imu_ex, au_ex, imu_au, au_au）

            # 按人名组织文件
            if person_name not in data_files:
                data_files[person_name] = {'imu_ex': [], 'au_ex': [], 'imu_au': [], 'au_au': []}
            if file_type in data_files[person_name]:
                data_files[person_name][file_type].append(file_name)
    return data_files

def general_dataset(user_mode, dataset_type, total_people, spilt_mode='cut', data_root='./data/',
                     dataset_root='./dataset/',window_length=46,step_length=12,
                     test_rate=0.2, val_rate=0.1,desired_length=1200):
    # if user_mode == "within":
    #     data_root = " "
    # elif user_mode == 'cross':
    #     data_root = " "
    # else:
    #     print("data mode wrong!")
    assert os.path.exists(data_root), "data path:{} does not exists".format(data_root)
    random.seed(0)  # 保证随机结果可复现

    data_files = dict_data(data_root)
    if user_mode == 'within':

        # 对文件进行配对和处理
        for person, files in data_files.items():
            person_dataset_path = '{}{}_P{}_{}_{}_w{}_s{}_{}_{}.pkl'.format(
                dataset_root,
                dataset_type,
                total_people,
                user_mode,
                spilt_mode,
                window_length,
                step_length,
                person,
                desired_length
            )
            
            if person != 'wzf':
                continue
            if os.path.exists(person_dataset_path):
                continue
            print("person",person)
            dataset_x = []
            dataset_y = []
            dataset_au = []
            # 获取此人的所有文件列表
            imu_ex_files = files['imu_ex']
            au_ex_files = files['au_ex']
            imu_au_files = files['imu_au']
            au_au_files = files['au_au']
            # 配对imu_ex和au_ex文件
            for imu_ex_file in imu_ex_files:
                # 构造对应的au_ex文件名
                au_ex_file = imu_ex_file.replace('imu_ex', 'au_ex')
                if au_ex_file in au_ex_files:
                    # read
                    personal_x, personal_y,personal_au = slidingWindow(os.path.join(data_root, imu_ex_file), os.path.join(data_root, au_ex_file),
                                                           window_length=window_length,step_length=step_length,
                                                           desired_length=desired_length)
                    # 添加到数据集
                    dataset_x.extend(personal_x)
                    dataset_y.extend(personal_y)
                    dataset_au.extend(personal_au)
                else:
                    print("au_ex_file is not matched in imu_ex_file!")
            for imu_au_file in imu_au_files:
                au_au_file = imu_au_file.replace('imu_au', 'au_au')
                if au_au_file in au_au_files:
                    personal_x, personal_y,personal_au = slidingWindow(os.path.join(data_root, imu_au_file), os.path.join(data_root, au_au_file),
                                                           window_length=window_length,step_length=step_length,
                                                           desired_length=desired_length)
                    # 添加到数据集
                    dataset_x.extend(personal_x)
                    dataset_y.extend(personal_y)
                    dataset_au.extend(personal_au)
                else:
                    print("au_au_file is not matched in imu_au_file!")
            # spilt dataset to train and test
            if spilt_mode == "random":
                val_size = int(len(dataset_y) * val_rate)
                test_size = int(len(dataset_y) * test_rate)
                # 创建索引列表
                indices = list(range(len(dataset_y)))

                # 随机选择验证集和测试集的索引
                val_indices = random.sample(indices, k=val_size)
                remaining_indices = list(set(indices) - set(val_indices))
                test_indices = random.sample(remaining_indices, k=test_size)

                # 剩余的索引用于训练集
                train_indices = list(set(remaining_indices) - set(test_indices))

                # 根据索引提取数据和标签
                x_train = [dataset_x[i] for i in train_indices]
                y_train = [dataset_y[i] for i in train_indices]
                au_train = [dataset_au[i] for i in train_indices]

                x_val = [dataset_x[i] for i in val_indices]
                y_val = [dataset_y[i] for i in val_indices]
                au_val = [dataset_au[i] for i in val_indices]

                x_test = [dataset_x[i] for i in test_indices]
                y_test = [dataset_y[i] for i in test_indices]
                au_test = [dataset_au[i] for i in test_indices]

                xy_dict = {'train_data': np.array(x_train), 'train_label': np.array(y_train), 'train_au': np.array(au_train),
                        'test_data': np.array(x_test), 'test_label': np.array(y_test), 'test_au': np.array(au_test),
                        'val_data': np.array(x_val), 'val_label': np.array(y_val), 'val_au': np.array(au_val)}
                with open(person_dataset_path, 'wb') as f:
                    pickle.dump(xy_dict, f, protocol=4)
            elif spilt_mode == "cut":
                dataset_length = len(dataset_y)
                num_train = int(dataset_length * 0.7)
                num_test = int(dataset_length * 0.2)
                num_vali = dataset_length - num_train - num_test

                border1s = [0, num_train, dataset_length - num_test]
                border2s = [num_train, num_train + num_vali, dataset_length]

                xy_dict = {'train_data':  np.array(dataset_x[border1s[0]:border2s[0]]), 
                           'train_label': np.array(dataset_y[border1s[0]:border2s[0]]), 
                           'train_au':    np.array(dataset_au[border1s[0]:border2s[0]]),
                           'test_data':   np.array(dataset_x[border1s[2]:border2s[2]]), 
                           'test_label':  np.array(dataset_y[border1s[2]:border2s[2]]),
                           'test_au':     np.array(dataset_au[border1s[2]:border2s[2]]),
                           'val_data':    np.array(dataset_x[border1s[1]:border2s[1]]),
                           'val_label':   np.array(dataset_y[border1s[1]:border2s[1]]),
                           'val_au':      np.array(dataset_au[border1s[1]:border2s[1]])
                           }
                # np.save(person_dataset_path, xy_dict)
                print(person_dataset_path, len(dataset_x), len(dataset_y))
                with open(person_dataset_path, 'wb') as f:
                    pickle.dump(xy_dict, f, protocol=4)
            # 
    elif user_mode == 'cross':
        for test_person in data_files.keys():
            person_dataset_path = '{}{}_P{}_{}_{}_w{}_s{}_{}_{}.pkl'.format(
                dataset_root,
                dataset_type,
                total_people,
                user_mode,
                spilt_mode,
                window_length,
                step_length,
                test_person,
                desired_length
            )
            if os.path.exists(person_dataset_path):
                continue
            # 数据集test：包含当前人名的数据
            test_data = []
            test_label = []
            test_au = []
            # 数据集train：包含除当前人名外的所有人的数据
            train_data = []
            train_label = []
            train_au = []
            if test_person != 'lls':
                continue

            for person, files in data_files.items():
                print("test_person and person :",test_person, person)
                # if os.path.exists(person_dataset_path):
                #     continue
                dataset_x = []
                dataset_y = []
                dataset_au = []
                # 获取此人的所有文件列表
                imu_ex_files = files['imu_ex']
                au_ex_files = files['au_ex']
                imu_au_files = files['imu_au']
                au_au_files = files['au_au']

                # 配对imu_ex和au_ex文件
                for imu_ex_file in imu_ex_files:
                    # 构造对应的au_ex文件名
                    au_ex_file = imu_ex_file.replace('imu_ex', 'au_ex')
                    if au_ex_file in au_ex_files:
                        # read
                        personal_x, personal_y, personal_au = slidingWindow(os.path.join(data_root, imu_ex_file), os.path.join(data_root, au_ex_file),
                                                            window_length=window_length,step_length=step_length,
                                                            desired_length=desired_length)
                        # 添加到数据集
                        dataset_x.extend(personal_x)
                        dataset_y.extend(personal_y)
                        dataset_au.extend(personal_au)
                    else:
                        print("au_ex_file is not matched in imu_ex_file!")
                for imu_au_file in imu_au_files:
                    au_au_file = imu_au_file.replace('imu_au', 'au_au')
                    if au_au_file in au_au_files:
                        personal_x, personal_y, personal_au = slidingWindow(os.path.join(data_root, imu_au_file), os.path.join(data_root, au_au_file),
                                                            window_length=window_length,step_length=step_length,
                                                            desired_length=desired_length)
                        # 添加到数据集
                        dataset_x.extend(personal_x)
                        dataset_y.extend(personal_y)
                        dataset_au.extend(personal_au)
                    else:
                        print("au_au_file is not matched in imu_au_file!")
                # save dataset
                if person == test_person:
                    test_data = dataset_x
                    test_label = dataset_y
                    test_au = dataset_au
                else:
                    train_data.extend(dataset_x)
                    train_label.extend(dataset_y)
                    train_au.extend(dataset_au)
            if spilt_mode == "random":
                val_size = int(len(train_label) * test_rate) # 只有train分出val，采用test_rate
                # 创建索引列表
                indices = list(range(len(train_label)))

                # 随机选择验证集和测试集的索引
                val_indices = random.sample(indices, k=val_size)
                train_indices = list(set(indices) - set(val_indices))

                # 根据索引提取数据和标签
                x_train = [train_data[i] for i in train_indices]
                y_train = [train_label[i] for i in train_indices]
                au_train = [train_au[i] for i in train_indices]

                x_val = [train_data[i] for i in val_indices]
                y_val = [train_label[i] for i in val_indices]
                au_val = [train_au[i] for i in val_indices]

            elif spilt_mode == "cut":
                dataset_length = len(train_label)
                num_train = dataset_length -int(dataset_length * 0.2)
                # 分离训练集、验证集和测试集的索引
                x_train = train_data[:num_train]
                y_train = train_label[:num_train]
                au_train = train_au[:num_train]

                x_val = train_data[num_train:]
                y_val = train_label[num_train:]
                au_val = train_au[num_train:]

            xy_dict = {'train_data': np.array(x_train),
                       'train_label': np.array(y_train),
                       'train_au': np.array(au_train),
                       'test_data': np.array(test_data),
                       'test_label': np.array(test_label),
                       'test_au': np.array(test_au),
                       'val_data': np.array(x_val),
                       'val_label': np.array(y_val),
                       'val_au': np.array(au_val)}
            # np.save(person_dataset_path, xy_dict)
            print(person_dataset_path, len(x_train), len(test_data),len(x_val))
            with open(person_dataset_path, 'wb') as f:
                pickle.dump(xy_dict, f, protocol=4)




if __name__ == '__main__':
    general_dataset(user_mode='within',
                    dataset_type='auex',
                    total_people=14,
                    spilt_mode='cut',
                    test_rate=0.2,
                    val_rate=0.1,
                    window_length=145, # au
                    step_length=2,
                    data_root='../dataset_testwzf/',
                    dataset_root='./dataset/',
                    desired_length=200) # imu



# class GenerateDataset(Dataset):
#     def __init__(self, root_path = './dataset/', window_length:int = 46,
#                  step_length:int = 23):
        
#         self.window_length = window_length
#         self.step_length = step_length
#         self.root_path = root_path
#         self.generateDataset()

#     def find_index(self, target, imu_time):
#         index = 0
#         while index <= len(imu_time):
#             if imu_time[index] >= target:
#                 break
#             index += 1
#         return index
#     # 切割数据
#     def slidingWindow(self, imu_data, au_data):

#         X = []
#         Y = []

#         # 从AU数据开始滑动窗口
#         for start_index in range(0, len(au_data), self.step_length):

#             end_index = start_index + self.window_length
#             if end_index > len(au_data):
#                 break

#             # 获取当前窗口的AU数据
#             current_au_window = au_data[start_index:end_index]

#             # 获取对应的IMU数据段
#             current_imu_segment = imu_data[self.find_index(current_au_window[0,0],imu_data[:,0]):self.find_index(current_au_window[-1,0],imu_data[:,0]),:]
#             # current_imu_segment = imu_data[(imu_data[:,0] >= current_au_window[0,0]) & (imu_data[:,0] <= current_au_window[-1,0])]
#             # print("imu segment length ",current_imu_segment.shape)

#             # 检查IMU数据长度，执行删除或插值
#             desired_length = 1200
#             if current_imu_segment.shape[0] < desired_length:
#                 time_stamps = imu_data[:, 0]
#                 new_time_stamps = np.linspace(time_stamps[0], time_stamps[-1], desired_length)

#                 new_imu_data = []
#                 for i in range(1, imu_data.shape[1]):
#                     # 对每列数据进行插值
#                     interpolator = interp1d(time_stamps, imu_data[:, i], kind='linear')
#                     new_column = interpolator(new_time_stamps)
#                     new_imu_data.append(new_column)
#                 new_imu_data = np.transpose(np.array(new_imu_data))
#             elif current_imu_segment.shape[0] > desired_length:
#                 # 裁剪数据,取中间的部分
#                 start_index = int((current_imu_segment.shape[0] - desired_length) / 2)
#                 end_index = start_index + desired_length
#                 new_imu_data = current_imu_segment[start_index:end_index, 1:]
#                 new_imu_data = np.array(new_imu_data)
#             else:
#                 new_imu_data = current_imu_segment[:, 1:]
#                 new_imu_data = np.array(new_imu_data)
            
#             new_au_data = current_au_window[:, 1:]
#             # print("new_imu_data",np.array(new_imu_data).shape)
#             # print("new_au_data",np.array(new_au_data).shape)

#             X.append(new_imu_data)
#             Y.append(new_au_data)
#         return X, Y
        
#     # 整合成一个数据集
#     def generateDataset(self):
#         # 所有的数据集
#         self.dataset_x = []
#         self.dataset_y = []
#         imu_path = os.path.join(self.root_path, "imu")
#         for file_name in os.listdir(imu_path):
#             file_name_parts = file_name.split('_')
#             if len(file_name_parts) >= 3:
#                 extracted_text = "au_" + file_name_parts[1] + '_' + file_name_parts[2]
#                 # print("extracted_text",extracted_text)
                
#             imu_data = pd.read_csv(os.path.join(imu_path, file_name), header=None).values
#             au_data = pd.read_csv(os.path.join(self.root_path, "au", extracted_text), header=None).values
#             # 切割数据
#             personal_x, personal_y = self.slidingWindow(imu_data, au_data)
#             print("single dataset ", file_name_parts[2] ,len(personal_x), len(personal_y))
#             # 添加到数据集
#             self.dataset_x.extend(personal_x)
#             self.dataset_y.extend(personal_y)
#         print("dataset_x shape:", len(self.dataset_x))



#     def readData(self, flag = 'train'):
#         # 根据类别分割数据集
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         set_type = type_map[flag]
#         dataset_length = len(self.dataset_x)
#         num_train = int(dataset_length * 0.7)
#         num_test = int(dataset_length * 0.2)
#         num_vali = dataset_length - num_train - num_test

#         border1s = [0, num_train, dataset_length - num_test]
#         border2s = [num_train, num_train + num_vali, dataset_length]
#         border1 = border1s[set_type]
#         border2 = border2s[set_type]
#         self.data_x = self.dataset_x[border1:border2]
#         self.data_y = self.dataset_y[border1:border2]

#     def __getitem__(self, index):
#         self.data_x[index] = np.array(self.data_x[index])
#         self.data_y[index] = np.array(self.data_y[index])

#         return self.data_x[index], self.data_y[index]

#     def __len__(self):
#         return len(self.data_x)
    
