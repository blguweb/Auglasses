
"""
@Description :  统计数据集中数据分布
@Author      :  kidominox 
@Time        :   2024/01/17 15:17:47
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class DatasetCDF():
    def __init__(self,data_path):
        self.data_path = data_path
        self.data = None
    
    def loadDataset(self):
        # 遍历文件下非文件夹的文件
        for filename in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, filename)
            # 检查是否是文件
            if os.path.isfile(file_path):
                f_data = pd.read_csv(file_path,header = None)
                f_data = f_data.values[:,1:]
                # 提取
                if self.data is None:
                    self.data = f_data
                else:
                    self.data = np.concatenate((self.data,f_data),axis=0)
        print(self.data.shape)

    def plotCDF(self):
        # 计算每个样本的最大特征值
        max_values_per_sample = self.data.max(axis=1)

        # 生成多个阈值
        thresholds = np.linspace(0, 5, 100)

        # 计算每个阈值的累积分布
        cdf_values = [np.mean(max_values_per_sample < threshold) for threshold in thresholds]

        # 绘制CDF曲线
        plt.plot(thresholds, cdf_values)
        plt.xlabel('Threshold')
        plt.ylabel('Proportion of samples with any feature < Threshold')
        plt.title('CDF of Max Feature Value per Sample')
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    data_path = "./dataset_all/au/"
    dataset = DatasetCDF(data_path)
    dataset.loadDataset()
    dataset.plotCDF()