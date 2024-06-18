import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
expression_list = ["happy","frown","open mouth"]
def spectrum_caculate():
    # 读取npy文件
    calibra_signal = np.load("calibra_signal.npy", allow_pickle=True)
    # 将加载的数据转换为字典
    calibra_signal = calibra_signal.item()

    sampling_rate = 400  # 采样频率为400 Hz
    channel_count = 18  # 通道数

    # 频率范围
    low_freq = 0  # 低频范围
    high_freq = 20  # 高频范围

    # 零填充的目标长度，根据需要进行调整
    target_length = 10000  # 假设您希望每个通道的长度扩展到10000个采样点

    # 循环字典的value，计算并绘制时频图
    
    for j in range(3):
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))  # 创建3x4的子图，用于前12列的时频图
        calibra_data = calibra_signal[expression_list[j]]  # 获取信号数据

        for i in range(12):  # 遍历前12列
            channel_data = calibra_data[:,i]
            original_length = len(channel_data)

            # 零填充，将信号长度扩展到目标长度
            zero_padded_data = np.pad(channel_data, (0, target_length - original_length), mode='constant')
            time_points = np.arange(len(zero_padded_data)) / sampling_rate  # 时间点数组

            # 进行FFT变换
            fft_result = np.fft.fft(zero_padded_data)
            freq_axis = np.fft.fftfreq(len(zero_padded_data), 1 / sampling_rate)

            # 获取在0-20 Hz范围内的频率索引
            low_freq_index = np.where(freq_axis >= low_freq)[0][0]
            high_freq_index = np.where(freq_axis <= high_freq)[0][-1]

            # 绘制时频图
            ax = axes[i // 3, i % 3]
            # ax.plot(freq_axis, np.abs(fft_result))
            ax.specgram(channel_data, Fs=sampling_rate, cmap='viridis', NFFT=64, noverlap=32,pad_to=10000)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Channel {i + 1}')
            # ax.set_xlim(0, len(zero_padded_data) / sampling_rate)
            ax.set_ylim(low_freq, high_freq)

        # 调整子图之间的距离
        plt.tight_layout()
        plt.show()


def spectrum_caculate2():
    # 读取npy文件
    calibra_signal = np.load("calibra_signal.npy", allow_pickle=True)
    # 将加载的数据转换为字典
    calibra_signal = calibra_signal.item()

    sampling_rate = 400  # 采样频率为400 Hz
    channel_count = 18  # 通道数

    # 循环字典的value，计算并绘制时频图
    
    for j in range(3):
        calibra_data = calibra_signal[expression_list[j]]  # 获取信号数据

        for i in range(12):  # 遍历前12列
            channel_data = calibra_data[:,i]
            print("channel_data.shape:",channel_data.shape)
            # Compute the spectrogram
            frequencies, times, spectrogram = signal.spectrogram(channel_data, fs=sampling_rate, nfft=10000, noverlap=32, nperseg=64, mode='magnitude')
            print(spectrogram.shape,type(spectrogram[4,3]))

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
            spectrogram_3d = np.repeat(spectrogram_resized_array[..., np.newaxis], 3, axis=2)

            # Visualizing the resized extracted spectrogram
            plt.figure(figsize=(6, 6))
            plt.imshow(spectrogram_resized_array, aspect='auto', cmap='hot')
            plt.title('Resized Extracted Spectrogram (0-20 Hz)')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar(label='Intensity')
            plt.show()

        # # 调整子图之间的距离
        # plt.tight_layout()
        # plt.show()



if __name__ == '__main__':
    spectrum_caculate2()