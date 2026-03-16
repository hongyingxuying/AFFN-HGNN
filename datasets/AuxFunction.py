# -*- coding: utf-8 -*-
"""
@author: HanJinZhe
@project: GNNforIFD
@file: AuxFunction.py
@time: 2022/11/24 23:25
@description： 
"""
import librosa
import numpy as np
from scipy import signal
import pywt



# 做快速傅里叶变换
def FFT(x):
    # x为原始信号
    x = np.fft.fft(x)
    x = np.abs(x) / len(x)
    x = x[range(int(x.shape[0]/2))]
    return x

# 添加噪声信号
def add_nosie(x,snr):
    # x为原始信号，snr为信噪比
    d = np.random.randn(len(x))
    p_signal = np.sum(abs(x) ** 2)
    p_d = np.sum(abs(d) ** 2)
    p_nosie = p_signal / 10 ** (snr / 10)
    noise = np.sqrt(p_nosie / p_d) * d
    noise_signal = x.reshape(-1) + noise
    return noise_signal

def wavelet_transform(x):
    """
    对一维振动信号进行小波变换。
    :param x: 输入的一维振动信号数组。
    :param wavelet: 使用的小波类型，默认为'db4'。
    :param level: 小波分解的级别，默认为1。
    :return: 小波变换的系数。
    """
    wavelet = 'db4'
    level = 1
    # 计算最大可能的分解级别
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)

    # 如果指定的级别大于最大级别，则使用最大级别
    level = min(level, max_level)

    # 执行小波分解
    coeffs = pywt.wavedec(x, wavelet, level=level)

    return coeffs

# 小波包变换
def wavelet_packet_transform(x):
    wavelet = 'db1'
    level = 3
    # Initialize the WaveletPacket object
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode='symmetric')

    # Determine the maximum level if not provided
    if level is None:
        level = wp.maxlevel

    # Decompose the signal
    wp.decompose()

    # Collect and return the nodes
    nodes = {node.path: node.data for node in wp.get_level(level, 'natural')}
    # return nodes
    flattened_data = []
    for path, data in sorted(nodes.items()):
        flattened_data.extend(data)

    # return nodes

    return np.array(flattened_data)


def emd(x):

    # 计算MFCC特征矩阵
    mfccs = librosa.feature.mfcc(y=x, sr=12000, n_mfcc=13)

    # 将MFCC特征矩阵的每一列拼接起来形成一个一维数组
    # 这里我们使用ravel方法，它默认按行顺序(flatten)来拼接矩阵，因此我们先转置矩阵
    flattened_features = mfccs.T.ravel()

    print("输出数组的长度：", len(flattened_features))
    # 输出一维数组
    return flattened_features
    # 输出数组长度



def stft_transform(x):
    """
    对一维振动信号进行短时傅里叶变换。
    :param x: 一维振动信号数组。
    :param fs: 采样频率，默认为 1.0。
    :param window: 用于 STFT 的窗口类型，默认为 'hann'。
    :param nperseg: 每个段的长度，默认为 256。
    :param noverlap: 重叠的点数，默认为 128。
    :return: 返回频率、时间和 STFT 结果的复数数组。
    """
    fs = 1.0
    window = 'hann'
    nperseg = 256
    noverlap = 128
    f, t, x_sft = signal.stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return x_sft

# 示例：假设您有一个信号
# fs = 采样频率
# signal_data = 振动信号数组
# f, t, Zxx = stft_transform(signal_data, fs=fs)

# 可视化 STFT
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
