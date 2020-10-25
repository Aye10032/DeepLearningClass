import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np

# 导入数据集、数据集乱序
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.shuffle(x_train)
np.random.shuffle(y_train)


# 定义神经网络结构
class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())  # 神经元个数、激活函数、正则化方法

    # 前向传播定义
    def call(self, inputs, training=None, mask=None):
        y = self.d1(inputs)
        return y


model = IrisModel()

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.1),  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数（已经过概率化分布）
    metrics=['sparse_categorical_accuracy']
)

# 训练
# validation_split：选择20%的数据作为测试集
# validation_freq：迭代多少次后在测试集中验证准确率
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()
