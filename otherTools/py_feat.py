from feat import Detector
import pandas as pd
import numpy as np
from feat.plotting import plot_face
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
# 创建面部检测器实例
detector = Detector()
print(detector)
# 指定视频文件路径
video_path = './dataCollection/video/ex_yh2.mp4'

# 从视频中提取面部特征
video_prediction  = detector.detect_video(video_path,antialiasing=True)
# print(video_prediction.head())
print(video_prediction.shape, type(video_prediction))
# 将提取的数据转换为 DataFrame
df = pd.DataFrame(video_prediction)

# 可视化
# video_prediction.loc[[0]].plot_detections(faceboxes=False, add_titles=False)
# axes = video_prediction.emotions.plot()
# extract labels and features


# au_columns = [col for col in video_prediction.columns if col.startswith('AU')]
# print(au_columns)

# x_train = video_prediction[au_columns].values.tolist()
# print(len(x_train),x_train)


#plot faces
def AU2tensor(AUs):
    muscles = {'all':'heatmap'}
    try:
        ax = plot_face(au=np.array(AUs), muscles = muscles)
    except ValueError as e:
        print(f"An error occurred: {e}")  # 打印异常的简单描述
        print('ValueError!! AUs are ',AUs)
        return np.NaN
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig = ax.get_figure()
    # plt.show()
    #fig.set_size_inches(400/fig.dpi,400/fig.dpi)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    plot_np = np.array(canvas.buffer_rgba())[:,:,:3].transpose((2,0,1))
    # tensor = torch.from_numpy(plot_np)
    plt.close()
    
    print('processed one plot')
    return plot_np

# tensor_train = [AU2tensor(x) for x in x_train]

# 保存数据到 CSV 文件
csv_path = './ex_yh2.csv'
df.to_csv(csv_path, index=False)

print(f"Face AU data extracted and saved to {csv_path}")


# 查看pyfeat的版本
# from feat.plotting import plot_face
# import numpy as np
# import matplotlib.pyplot as plt
# # 20 dimensional vector of AU intensities
# # AUs ordered as:
# # 1, 2, 4, 5, 6, 7, 9, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43
# neutral = np.zeros(20)

# ax = plot_face(au=neutral, title='Neutral')
# plt.show()