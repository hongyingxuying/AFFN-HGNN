# -*- coding: utf-8 -*-
"""
@author: HanJinZhe
@project: GNNforIFD
@file: relitu.py
@time: 2024/3/13 14:57
@description： 
"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Matrix = [[1, 0, 0, 0, 0], [0, 0.94461538, 0.00307692, 0.05230769, 0],
          [0, 0.00327869, 0.95737705, 0.03934426, 0], [0, 0.00847458, 0.00677966, 0.98474576, 0],
          [0, 0, 0, 0, 1]]

# 绘制热力图
plt.figure(figsize=(10, 7))
sns.heatmap(Matrix, annot=True, fmt=".2%",cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
