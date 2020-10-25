import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()

# print('x_train.shape:\n', x_train.shape)
# print('y_train.shape:\n', y_train.shape)
# print('x_test.shape:\n', x_test.shape)
# print('y_test.shape:\n', y_test.shape)
#
# plt.imshow(x_train[0], cmap='gray')
# plt.show()

x_train, x_test = x_train / 255.0, x_test / 255.0  # 输入特征归一化，将0-255变为0-1

tbCallBack = TensorBoard(log_dir='./logs')  # log 目录


# histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
# batch_size=32,  # 用多大量的数据计算直方图
# write_graph=True,  # 是否存储网络结构图
# write_grads=True,  # 是否可视化梯度直方图
# write_images=True,  # 是否可视化参数
# embeddings_freq=0,
# embeddings_layer_names=None,
# embeddings_metadata=None)


# class FashionModel(Model):
#     def __init__(self):
#         super(FashionModel, self).__init__()
#         self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
#         self.d1 = tf.keras.layers.Dense(32, activation='relu')
#         self.d2 = tf.keras.layers.Dropout(0.2)
#         self.d3 = tf.keras.layers.Dense(10, activation='softmax')
#
#     def call(self, inputs, training=None, mask=None):
#         inputs = self.flatten(inputs)
#         inputs = self.d1(inputs)
#         inputs = self.d2(inputs)
#         outputs = self.d3(inputs)
#
#         return outputs
#
#
# model = FashionModel()

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(
    optimizer='adam',  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数（已经过概率化分布）
    metrics=['sparse_categorical_accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1,
          callbacks=[tbCallBack])
model.summary()
