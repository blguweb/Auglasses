from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
import pickle

class My_Dataset(Dataset):
    """自定义数据集"""

    def __init__(self, path, label):
        # self.images_path = path
        # 加载.npy文件
        # dict_dataset = np.load(path, allow_pickle=True)
        # dict_dataset = dict_dataset.item()
        with open(path, 'rb') as f:
            dict_dataset = pickle.load(f)

        if label == 'train':
            self.x = dict_dataset["train_data"]
            self.y = dict_dataset["train_label"]

        elif label == 'test':
            self.x = dict_dataset['test_data']
            self.y = dict_dataset['test_label']

        elif label =='val':
            self.x = dict_dataset['val_data']
            self.y = dict_dataset['val_label']


    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        # tensor_ = torch.from_numpy(self.x[item]).float()
        # labels = torch.from_numpy(self.y[item]).float()
        # au = torch.from_numpy(self.fir_au[item]).float()
        #
        
        # label = np.delete(self.y[item],(5,9,11,16), axis=1)

        # y = torch.from_numpy(y).float()
        # grade
        return self.x[item], self.y[item]

