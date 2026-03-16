# -*- coding: utf-8 -*-
"""
@author: HanJinZhe
@project: GNNforIFD
@file: PUPath.py
@time: 2022/12/6 21:24
@description： 
"""
import os

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.PathGraph import pathGraph
from datasets.RadiusGraph import RadiusGraph
from datasets.KNNGraph import KNNGraph
from datasets.AuxFunction import FFT, stft_transform, wavelet_packet_transform
import pickle

from tqdm import tqdm

signal_size = 2048
root = r'data/PU'

# 1 Undamaged (healthy) bearings(6X)
# HBdata = ['K001', "K002", 'K003', 'K004', 'K005', 'K006']
HBdata = ['K001']
# label1 = [0, 1, 2, 3, 4, 5]  # The undamaged (healthy) bearings data is labeled 1-9
label1 = [0]  # The undamaged (healthy) bearings data is labeled 1-9
# 2 Artificially damaged bearings(12X)
ADBdata = ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03', 'KI05', 'KI07', 'KI08']
label2 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # The artificially damaged bearings data is labeled 4-15
# 3 Bearings with real damages caused by accelerated lifetime tests(14x)
# RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI04','KI14','KI16','KI17','KI18','KI21']
# label3=[18,19,20,21,22,23,24,25,26,27,28,29,30,31]  #The artificially damaged bearings data is labeled 16-29
RDBdata = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
# RDBdata = ['KA16', 'KI16']

label3 = [i for i in range(13)]
# label3 = [1, 2]

# working condition
WC = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
state = WC[0]  # WC[0] can be changed to different working states


# generate Training Dataset and Testing Dataset
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
    fl1 = fl[0][0][2][0][6][2]  # Take out the data
    fl2 = fl[0][0][2][0][5][2] # 扭矩
    fl1 = (fl1 - fl1.min()) / (fl1.max() - fl1.min())
    fl2 = (fl2 - fl2.min()) / (fl2.max() - fl2.min())
    fl1 = fl1.reshape(-1, )
    fl2 = fl2.reshape(-1, )
    datax = []
    datadl = []
    start, end = 0, signal_size
    while end <= fl1[:signal_size * 1000].shape[0]:
        if input_type == "TD":
            x = fl1[start:end]
        elif input_type == "FD":
            x = fl1[start:end]
            # x_fd = FFT(x)
            # x_wpt = wavelet_packet_transform(x)
            datax.append(x)
            # data_combined.append(x_wpt)
        else:
            print("The input_type is wrong!!")

        # data1.append(x)

        start += signal_size
        end += signal_size

    while end <= fl2[:signal_size * 1000].shape[0]:
        if input_type == "TD":
            x = fl2[start:end]
        elif input_type == "FD":
            xdl = fl2[start:end]
            # x_fd = FFT(x)
            # x_wpt = wavelet_packet_transform(x)
            datadl.append(xdl)
            # data_combined.append(x_wpt)
        else:
            print("The input_type is wrong!!")

        # data1.append(x)

        start += signal_size
        end += signal_size

    # graphset1 = pathGraph(20, data_combined, label, task)
    # graphset3 = RadiusGraph(20, data_combined, label, task)
    graphset1 = KNNGraph(20, datax, label, task)
    graphset2 = KNNGraph(20, datadl, label, task)
    # graphset = graphset1 + graphset2
    return graphset1,graphset2


class PUPathronghe(object):
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
            with open(os.path.join(self.data_dir, "PUPath.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if test:
            test_dataset = list_data
            return test_dataset
        else:
            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)

            return train_dataset, val_dataset


