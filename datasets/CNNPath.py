# -*- coding: utf-8 -*-
"""
@author: HanJinZhe
@project: GNNforIFD
@file: CNNPath.py
@time: 2024/3/9 13:16
@description： 
"""
import os

import pywt
from matplotlib import pyplot as plt
from scipy.io import loadmat
from datasets.PathGraph import pathGraph
from datasets.RadiusGraph import RadiusGraph
from datasets.KNNGraph import KNNGraph
from datasets.AuxFunction import FFT, stft_transform, wavelet_packet_transform
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib import cm

signal_size = 1024
root = r'data/PU'

# RDBdata = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
RDBdata = ['KA04', 'KA15']
# label3 = [i for i in range(13)]
label3 = [0, 1]
WC = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
state = WC[0]


def get_files(sample_length, root, input_type, task, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    #
    # for i in tqdm(range(len(HBdata))):
    #     name1 = state+"_"+HBdata[i]+"_1"
    #     path1=os.path.join(root,HBdata[i],name1+".mat")        #_1----->1 can be replaced by the number between 1 and 20
    #     data1 = data_load(sample_length, path1,name=name1,label=label1[i],input_type=input_type,task=task)
    #     data += data1

    #
    # for j in tqdm(range(len(ADBdata))):
    #     name2 = state+"_"+ADBdata[j]+"_1"
    #     path2=os.path.join(root,ADBdata[j],name2+".mat")
    #     data2 = data_load(sample_length, path2,name=name2,label=label2[j],input_type=input_type,task=task)
    #     data += data2

    for k in tqdm(range(len(RDBdata))):
        name3 = state + "_" + RDBdata[k] + "_1"
        path3 = os.path.join(root, RDBdata[k], name3 + ".mat")
        data3 = data_load(sample_length, path3, name=name3, label=label3[k], input_type=input_type, task=task)
        data += data3

    return data


def data_load(signal_size, filename, name, label, input_type, task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    name:在使用loadmat函数时，需要提供MATLAB文件的名称作为参数，并通过字典键名来访问其中的数据。loadmat返回的是一个字典，其中包含了MATLAB文件中的所有变量，每个变量作为一个键值对保存在字典中，键名为MATLAB中变量的名称，对应的值为Python中的numpy数组或矩阵。
    fl[0][0][2][0][6][2]  # vibration data
    fl[0][0][2][0][3][2]  # speed data
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  # Take out the data
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1, )
    datax = []
    data1 = []  # 存储时域和频域数据的列表
    start, end = 0, signal_size
    while end <= fl[:signal_size * 1000].shape[0]:
        if input_type == "TD":
            x = fl[start:end]
        elif input_type == "FD":
            x = fl[start:end]
            # x_fd = FFT(x)
            # wavelet = 'db1'
            # maxlevel = np.min([pywt.dwt_max_level(data_len=len(x), filter_len=pywt.Wavelet(wavelet).dec_len), 4])
            #
            # # 执行小波包变换
            # wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
            #
            # # 准备绘图
            # fig, ax = plt.subplots(figsize=(12, 8))
            #
            # # 获取所有叶子节点数据
            # leaf_nodes = wp.get_leaf_nodes(decompose=True)
            # values = np.array([node.data for node in leaf_nodes])
            #
            # # 绘制时频图
            # extent = [0, x.size, 0, len(leaf_nodes)]
            # im = ax.imshow(values, cmap='coolwarm', aspect='auto', extent=extent)
            #
            # # 设置图表标题和坐标轴标签
            # ax.set_title('Wavelet Packet Transform (WPT) Time-Frequency Representation')
            # ax.set_ylabel('Frequency Band')
            # ax.set_xlabel('Time')
            #
            # # 避免使用可能导致错误的频率计算方式
            # # 直接指定y轴刻度的位置和标签
            # ax.set_yticks(np.arange(0.5, len(leaf_nodes) + 0.5))
            # ax.set_yticklabels(["Band {}".format(i + 1) for i in range(len(leaf_nodes))])
            #
            # # 添加颜色条
            # fig.colorbar(im, ax=ax, orientation='vertical', label='Coefficient magnitude')
            #
            # plt.show()

            wavelet = 'db1'
            maxlevel = np.min([pywt.dwt_max_level(data_len=len(x), filter_len=pywt.Wavelet(wavelet).dec_len), 3])

            # 执行小波包变换
            wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)

            # 准备绘图
            # fig, ax = plt.subplots(figsize=(12, 8))

            # 获取所有叶子节点数据
            leaf_nodes = wp.get_leaf_nodes(decompose=True)
            values = np.array([node.data for node in leaf_nodes])

            # 确保values归一化到0-1
            values_normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
            print(values[0])
            # 将归一化的值映射到0-255范围，并转换为整数
            gray_values = np.rint(values_normalized * 255).astype(int)

            # 将结果存储到列表中
            # gray_values_list = gray_values

            for val in gray_values:
                datax.append(val)

            # 绘制时频图
            # extent = [0, x.size, 0, len(leaf_nodes)]
            # im = ax.imshow(values, cmap='coolwarm', aspect='auto', extent=extent)
            #
            # # 不显示图像
            # plt.close(fig)
            # 设置图表标题和坐标轴标签
            # ax.set_title('Wavelet Packet Transform (WPT) Time-Frequency Representation')
            # ax.set_ylabel('Frequency Band')
            # ax.set_xlabel('Time')
            #
            # # 避免使用可能导致错误的频率计算方式
            # # 直接指定y轴刻度的位置和标签
            # ax.set_yticks(np.arange(0.5, len(leaf_nodes) + 0.5))
            # ax.set_yticklabels(["Band {}".format(i + 1) for i in range(len(leaf_nodes))])
            #
            # # 添加颜色条
            # fig.colorbar(im, ax=ax, orientation='vertical', label='Coefficient magnitude')
            #
            # plt.show()

            # 将颜色值转换为灰度值
            # 使用matplotlib的colormap和归一化函数转换值
            # norm = plt.Normalize(vmin=values.min(), vmax=values.max())
            # 使用coolwarm颜色图
            # mappable = cm.ScalarMappable(norm=norm, cmap='coolwarm')
            # 将values数组中的值转换为RGBA颜色，然后取平均值转换为灰度值
            # gray_values = np.mean(mappable.to_rgba(values)[:, :, :3], axis=2)
            # print(type(gray_values))
            # 存储灰度值
            # datax = gray_values.flatten()
            # for val in gray_values:
            #     datax.append(val)

            # print("111111111:", datax)
            # 如果需要输出datax中的值
            # print(datax.shape)

            # 将数据块转换为32x32的二维数组
            # x_reshaped = x.reshape(32, 32)
            # 将转换后的数据添加到datax列表中
            # data1.append(xwpt)
        else:
            print("The input_type is wrong!!")

        # data1.append(x)

        start += signal_size
        end += signal_size
    # 可选：可视化第一个转换后的灰度图，以验证转换过程
    # plt.imshow(datax[0], cmap='gray')
    # plt.colorbar()
    # plt.title("Example of Reshaped Vibration Signal as Grayscale Image")
    # plt.show()

    # 假设datax是一个n个元素的列表，每个元素是一个灰度图对应的特征向量
    # features = np.array(datax)  # 将列表转换为NumPy数组，便于处理

    # # 使用KNN找到每个节点的最近邻
    # k = 5  # 选择的最近邻个数
    # nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features)
    # distances, indices = nbrs.kneighbors(features)

    # 构建邻接矩阵
    # n = len(features)
    # adj_matrix = np.zeros((n, n))  # 初始化邻接矩阵
    #
    # for i in range(n):
    #     for j in indices[i][1:]:  # 跳过自己，从1开始
    #         adj_matrix[i][j] = 1  # 无权图，所以使用1表示连接

    # 计算边的权重（可选，如果你需要有权图）
    # 例如，使用距离的倒数作为权重
    # for i in range(n):
    #     for j, dist in zip(indices[i][1:], distances[i][1:]):
    #         adj_matrix[i][j] = 1 / (dist + 1e-5)  # 避免除以0
    # print(datax)
    graphset = pathGraph(10, datax, label, task)
    # print(graphset)
    # graphset = graphset1 + graphset2 + graphset3
    return graphset


class CNNPath(object):
    num_classes = 13

    def __init__(self, sample_length, data_dir, input_type, task):
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.input_type = input_type
        self.task = task

    def data_prepare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            list_data = get_files(self.sample_length, self.data_dir, self.input_type, self.task, test)
            with open(os.path.join(self.data_dir, "PUPath_wpt.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if test:
            test_dataset = list_data
            return test_dataset
        else:
            # print(list_data)
            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)

            return train_dataset, val_dataset
