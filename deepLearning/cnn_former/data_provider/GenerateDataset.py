import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import pickle
import lmdb
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
    # A = []
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
        # A.append(np.array(current_au_window))
    print(len(X),len(Y))
    return X, Y

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
            person_dataset_path = '{}{}_P{}_{}_{}_w{}_s{}_{}_{}'.format(
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
# 
            if person != 'yzh':
            # ['lyr','cyx' ,'dll', 'dyw', 'hyj', 'lls']:
                continue
            if os.path.exists(person_dataset_path):
                continue
            os.makedirs(os.path.dirname(person_dataset_path), exist_ok=True)
            lmdb_holder = lmdb.open(person_dataset_path, map_size=int(1099511627776))  # You might need to adjust the map_size based on your dataset size
            print("person",person)
            dataset_x_train = []
            dataset_x_test = []
            dataset_x_val = []
            dataset_y_train = []
            dataset_y_test = []
            dataset_y_val = []
            train_length = 0
            test_length = 0
            val_length = 0
            # dataset_au_train = []
            # dataset_au_test = []
            # dataset_au_val = []
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
                    personal_x, personal_y = slidingWindow(os.path.join(data_root, imu_ex_file), os.path.join(data_root, au_ex_file),
                                                           window_length=window_length,step_length=step_length,
                                                           desired_length=desired_length)
                    # 添加到数据集 cut 方式
                    dataset_length = len(personal_y)
                    num_train = int(dataset_length * 0.7)
                    num_test = int(dataset_length * 0.2)
                    num_vali = dataset_length - num_train - num_test

                    border1s = [0, num_train, dataset_length - num_test]
                    border2s = [num_train, num_train + num_vali, dataset_length]

                    for px_i in range(border1s[0],border2s[0]):
                        txn = lmdb_holder.begin(write=True)
                        train_data_key = f'train_data_{train_length}'.encode('utf-8')
                        train_label_key = f'train_label_{train_length}'.encode('utf-8')
                        txn.put(train_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                        txn.put(train_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                        train_length += 1
                        txn.commit()

                    for px_i in range(border1s[1],border2s[1]):
                        txn = lmdb_holder.begin(write=True)
                        val_data_key = f'val_data_{val_length}'.encode('utf-8')
                        val_label_key = f'val_label_{val_length}'.encode('utf-8')
                        txn.put(val_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                        txn.put(val_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                        val_length += 1
                        txn.commit()

                    for px_i in range(border1s[2],border2s[2]):
                        txn = lmdb_holder.begin(write=True)
                        test_data_key = f'test_data_{test_length}'.encode('utf-8')
                        test_label_key = f'test_label_{test_length}'.encode('utf-8')
                        txn.put(test_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                        txn.put(test_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                        test_length += 1
                        txn.commit()
                    # dataset_x_train.extend(personal_x[border1s[0]:border2s[0]])
                    # dataset_x_test.extend(personal_x[border1s[2]:border2s[2]])
                    # dataset_x_val.extend(personal_x[border1s[1]:border2s[1]])
                    # dataset_y_train.extend(personal_y[border1s[0]:border2s[0]])
                    # dataset_y_test.extend(personal_y[border1s[2]:border2s[2]])
                    # dataset_y_val.extend(personal_y[border1s[1]:border2s[1]])
                    # dataset_au_train.extend(personal_au[border1s[0]:border2s[0]])
                    # dataset_au_test.extend(personal_au[border1s[2]:border2s[2]])
                    # dataset_au_val.extend(personal_au[border1s[1]:border2s[1]])
                
                else:
                    print("au_ex_file is not matched in imu_ex_file!")
            for imu_au_file in imu_au_files:
                au_au_file = imu_au_file.replace('imu_au', 'au_au')
                if au_au_file in au_au_files:
                    personal_x, personal_y = slidingWindow(os.path.join(data_root, imu_au_file), os.path.join(data_root, au_au_file),
                                                           window_length=window_length,step_length=step_length,
                                                           desired_length=desired_length)
                    dataset_length = len(personal_y)
                    num_train = int(dataset_length * 0.7)
                    num_test = int(dataset_length * 0.2)
                    num_vali = dataset_length - num_train - num_test

                    border1s = [0, num_train, dataset_length - num_test]
                    border2s = [num_train, num_train + num_vali, dataset_length]

                    for px_i in range(border1s[0],border2s[0]):
                        txn = lmdb_holder.begin(write=True)
                        train_data_key = f'train_data_{train_length}'.encode('utf-8')
                        train_label_key = f'train_label_{train_length}'.encode('utf-8')
                        txn.put(train_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                        txn.put(train_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                        train_length += 1
                        txn.commit()

                    for px_i in range(border1s[1],border2s[1]):
                        txn = lmdb_holder.begin(write=True)
                        val_data_key = f'val_data_{val_length}'.encode('utf-8')
                        val_label_key = f'val_label_{val_length}'.encode('utf-8')
                        txn.put(val_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                        txn.put(val_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                        val_length += 1
                        txn.commit()

                    for px_i in range(border1s[2],border2s[2]):
                        txn = lmdb_holder.begin(write=True)
                        test_data_key = f'test_data_{test_length}'.encode('utf-8')
                        test_label_key = f'test_label_{test_length}'.encode('utf-8')
                        txn.put(test_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                        txn.put(test_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                        test_length += 1
                        txn.commit()
                    # dataset_x_train.extend(personal_x[border1s[0]:border2s[0]])
                    # dataset_x_test.extend(personal_x[border1s[2]:border2s[2]])
                    # dataset_x_val.extend(personal_x[border1s[1]:border2s[1]])
                    # dataset_y_train.extend(personal_y[border1s[0]:border2s[0]])
                    # dataset_y_test.extend(personal_y[border1s[2]:border2s[2]])
                    # dataset_y_val.extend(personal_y[border1s[1]:border2s[1]])
                    # dataset_au_train.extend(personal_au[border1s[0]:border2s[0]])
                    # dataset_au_test.extend(personal_au[border1s[2]:border2s[2]])
                    # dataset_au_val.extend(personal_au[border1s[1]:border2s[1]])

                else:
                    print("au_au_file is not matched in imu_au_file!")
            # spilt dataset to train and test
            # np.save(person_dataset_path, xy_dict)

            # with open(person_dataset_path, 'wb') as f:
            #     pickle.dump(xy_dict, f, protocol=4)
            # Ensure the directory for LMDB exists

            print(person_dataset_path, train_length, test_length, val_length)

            txn = lmdb_holder.begin(write=True)
            txn.put( f'train_len'.encode('utf-8'), pickle.dumps(train_length, protocol=4))
            txn.put( f'test_len'.encode('utf-8'), pickle.dumps(test_length, protocol=4))
            txn.put( f'val_len'.encode('utf-8'), pickle.dumps(val_length, protocol=4))
            txn.commit()
            

            lmdb_holder.close()  # 关闭LMDB环境
            # 
    elif user_mode == 'cross':
        for test_person in data_files.keys():
            person_dataset_path = '{}{}_P{}_{}_{}_w{}_s{}_{}_{}'.format(
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
            if test_person != 'yzh':
                continue
            if os.path.exists(person_dataset_path):
                continue
            # Ensure the directory for LMDB exists
            os.makedirs(os.path.dirname(person_dataset_path), exist_ok=True)

            lmdb_holder = lmdb.open(person_dataset_path, map_size=int(1099511627776))  # You might need to adjust the map_size based on your dataset size

            # 数据集test：包含当前人名的数据
            # test_data = []
            # test_label = []
            test_length = 0
            # test_au = []
            # 数据集train：包含除当前人名外的所有人的数据
            # val
            # train_data = []
            # train_label = []
            train_length = 0
            # train_au = []
            # val_data = []
            # val_label = []
            val_length = 0
            # val_au = []


            for person, files in data_files.items():
                print("test_person and person :",test_person, person)
                # if os.path.exists(person_dataset_path):
                #     continue

                # 获取此人的所有文件列表
                imu_ex_files = files['imu_ex']
                au_ex_files = files['au_ex']
                imu_au_files = files['imu_au']
                au_au_files = files['au_au']
                if person == test_person:
                    # 没有重构QAQ
                    # 配对imu_ex和au_ex文件
                    for imu_ex_file in imu_ex_files:
                        # 构造对应的au_ex文件名
                        au_ex_file = imu_ex_file.replace('imu_ex', 'au_ex')
                        if au_ex_file in au_ex_files:
                            # read
                            personal_x, personal_y = slidingWindow(os.path.join(data_root, imu_ex_file), os.path.join(data_root, au_ex_file),
                                                                window_length=window_length,step_length=step_length,
                                                                desired_length=desired_length)
                            # 添加到数据集
                            for px_i in range(len(personal_y)):
                                txn = lmdb_holder.begin(write=True)
                                test_data_key = f'test_data_{test_length}'.encode('utf-8')
                                test_label_key = f'test_label_{test_length}'.encode('utf-8')
                                txn.put(test_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                                txn.put(test_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                                test_length += 1
                                txn.commit()
                            # test_data.extend(personal_x)
                            # test_label.extend(personal_y)
                            # test_au.extend(personal_au)
                        else:
                            print("au_ex_file is not matched in imu_ex_file!")
                    for imu_au_file in imu_au_files:
                        au_au_file = imu_au_file.replace('imu_au', 'au_au')
                        if au_au_file in au_au_files:
                            personal_x, personal_y = slidingWindow(os.path.join(data_root, imu_au_file), os.path.join(data_root, au_au_file),
                                                                window_length=window_length,step_length=step_length,
                                                                desired_length=desired_length)
                            # 添加到数据集
                            for px_i in range(len(personal_y)):
                                txn = lmdb_holder.begin(write=True)
                                test_data_key = f'test_data_{test_length}'.encode('utf-8')
                                test_label_key = f'test_label_{test_length}'.encode('utf-8')
                                txn.put(test_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                                txn.put(test_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                                test_length += 1
                                txn.commit()
                            # test_data.extend(personal_x)
                            # test_label.extend(personal_y)
                            # test_au.extend(personal_au)
                        else:
                            print("au_au_file is not matched in imu_au_file!")
            
                else:
                    # 配对imu_ex和au_ex文件
                    for imu_ex_file in imu_ex_files:
                        # 构造对应的au_ex文件名
                        au_ex_file = imu_ex_file.replace('imu_ex', 'au_ex')
                        if au_ex_file in au_ex_files:
                            # read
                            personal_x, personal_y = slidingWindow(os.path.join(data_root, imu_ex_file), os.path.join(data_root, au_ex_file),
                                                                window_length=window_length,step_length=step_length,
                                                                desired_length=desired_length)
                            dataset_length = len(personal_y)
                            num_train = dataset_length -int(dataset_length * 0.2)

                            # 添加到数据集
                            for px_i in range(num_train):
                                txn = lmdb_holder.begin(write=True)
                                train_data_key = f'train_data_{train_length}'.encode('utf-8')
                                train_label_key = f'train_label_{train_length}'.encode('utf-8')
                                txn.put(train_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                                txn.put(train_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                                train_length += 1
                                txn.commit()
                            for vx_i in range(num_train, dataset_length):
                                txn = lmdb_holder.begin(write=True)
                                val_data_key = f'val_data_{val_length}'.encode('utf-8')
                                val_label_key = f'val_label_{val_length}'.encode('utf-8')
                                txn.put(val_data_key, pickle.dumps(personal_x[vx_i],protocol=4))
                                txn.put(val_label_key, pickle.dumps(personal_y[vx_i],protocol=4))
                                val_length += 1
                                txn.commit()
                            # train_data.extend(personal_x[:num_train])
                            # train_label.extend(personal_y[:num_train])
                            # train_au.extend(personal_au[:num_train])
                            # val_data.extend(personal_x[num_train:])
                            # val_label.extend(personal_y[num_train:])
                            # val_au.extend(personal_au[num_train:])
                        else:
                            print("au_ex_file is not matched in imu_ex_file!")
                    for imu_au_file in imu_au_files:
                        au_au_file = imu_au_file.replace('imu_au', 'au_au')
                        if au_au_file in au_au_files:
                            personal_x, personal_y = slidingWindow(os.path.join(data_root, imu_au_file), os.path.join(data_root, au_au_file),
                                                                window_length=window_length,step_length=step_length,
                                                                desired_length=desired_length)
                            # 添加到数据集
                            dataset_length = len(personal_y)
                            num_train = dataset_length -int(dataset_length * 0.2)

                            # 添加到数据集
                            for px_i in range(num_train):
                                txn = lmdb_holder.begin(write=True)
                                train_data_key = f'train_data_{train_length}'.encode('utf-8')
                                train_label_key = f'train_label_{train_length}'.encode('utf-8')
                                txn.put(train_data_key, pickle.dumps(personal_x[px_i],protocol=4))
                                txn.put(train_label_key, pickle.dumps(personal_y[px_i],protocol=4))
                                train_length += 1
                                txn.commit()
                            for vx_i in range(num_train, dataset_length):
                                txn = lmdb_holder.begin(write=True)
                                val_data_key = f'val_data_{val_length}'.encode('utf-8')
                                val_label_key = f'val_label_{val_length}'.encode('utf-8')
                                txn.put(val_data_key, pickle.dumps(personal_x[vx_i],protocol=4))
                                txn.put(val_label_key, pickle.dumps(personal_y[vx_i],protocol=4))
                                val_length += 1
                                txn.commit()
                            # train_data.extend(personal_x[:num_train])
                            # train_label.extend(personal_y[:num_train])
                            # train_au.extend(personal_au[:num_train])
                            # val_data.extend(personal_x[num_train:])
                            # val_label.extend(personal_y[num_train:])
                            # val_au.extend(personal_au[num_train:])
                        else:
                            print("au_au_file is not matched in imu_au_file!")


            # xy_dict = {'train_data': np.array(train_data),
            #            'train_label': np.array(train_label),
            #            'train_au': np.array(train_au),
            #            'test_data': np.array(test_data),
            #            'test_label': np.array(test_label),
            #            'test_au': np.array(test_au),
            #            'val_data': np.array(val_data),
            #            'val_label': np.array(val_label),
            #            'val_au': np.array(val_au)}
            # np.save(person_dataset_path, xy_dict)
            print(person_dataset_path, train_length, test_length, val_length)
            # with open(person_dataset_path, 'wb') as f:
            #     pickle.dump(xy_dict, f, protocol=4)
            txn = lmdb_holder.begin(write=True)
            txn.put( f'train_len'.encode('utf-8'), pickle.dumps(train_length, protocol=4))
            txn.put( f'test_len'.encode('utf-8'), pickle.dumps(test_length, protocol=4))
            txn.put( f'val_len'.encode('utf-8'), pickle.dumps(val_length, protocol=4))
            txn.commit()
            # with lmdb_holder.begin(write=True) as txn:
            #     # 为每个数据集数组中的每个样本分配一个唯一的键并存储
            #     for dataset_name, dataset_array in [('train_data', train_data), 
            #                                         ('train_label', train_label), 
            #                                         ('test_data', test_data), 
            #                                         ('test_label', test_label),
            #                                         ('val_data', val_data),
            #                                         ('val_label', val_label)]:
            #         for index, value in enumerate(dataset_array):
            #             # 构造键名，例如 'val_au_123'
            #             key = f'{dataset_name}_{index}'.encode('utf-8')
            #             # 序列化值
            #             value_serialized = pickle.dumps(value, protocol=4)
            #             # 存储键值对
            #             txn.put(key, value_serialized)
                # 存储长度
            lmdb_holder.close()  # 关闭LMDB环境




if __name__ == '__main__':
    general_dataset(user_mode='within',
                    dataset_type='auex',
                    total_people=14,
                    spilt_mode='cut',
                    test_rate=0.2,
                    val_rate=0.1,
                    window_length=60, # au
                    step_length=1,
                    data_root='../300hz_data_auex/',
                    dataset_root='../datasetauex_300hz/',
                    desired_length=150) # imu
