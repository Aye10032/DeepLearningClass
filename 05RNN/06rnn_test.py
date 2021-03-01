import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

maotai = pd.read_csv('../data/SH600519.csv')

training_set = maotai.iloc[0:2426 - 300, 2:3]  # 0-2126行的第3列作为训练数据
test_set = maotai[2426 - 300:, 2:3]

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 归一化为0-1
training_set_scaled = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

x_train = []
y_train = []

x_test = []
y_test = []


