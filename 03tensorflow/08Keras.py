import os

import tensorflow as tf
from sklearn import datasets
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入数据集、数据集乱序
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.shuffle(x_train)
np.random.shuffle(y_train)

# 定义神经网络结构
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())  # 神经元个数、激活函数、正则化方法
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数（已经过概率化分布）
    metrics=['sparse_categorical_accuracy']
)

# 训练
# validation_split：选择20%的数据作为测试集
# validation_freq：迭代多少次后在测试集中验证准确率
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()
