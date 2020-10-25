import os
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0  # 输入特征归一化，将0-255变为0-1


class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.flatten(inputs)
        inputs = self.d1(inputs)
        outputs = self.d2(inputs)

        return outputs

    def get_config(self):
        pass


model = MnistModel()

model.compile(
    optimizer='adam',  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数（已经过概率化分布）
    metrics=['sparse_categorical_accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
