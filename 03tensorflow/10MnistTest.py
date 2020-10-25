import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plt.imshow(x_train[0], cmap='gray')
# plt.show()
# print('x_train[0]:\n', x_train[0])
# print('y_train[0]:\n', y_train[0])

# print('x_train.shape:\n', x_train.shape)
# print('y_train.shape:\n', y_train.shape)
# print('x_test.shape:\n', x_test.shape)
# print('y_test.shape:\n', y_test.shape)
# 输出：
# x_train.shape:
#  (60000, 28, 28)
# y_train.shape:
#  (60000,)
# x_test.shape:
#  (10000, 28, 28)
# y_test.shape:
#  (10000,)

x_train, x_test = x_train / 255.0, x_test / 255.0  # 输入特征归一化，将0-255变为0-1

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),  # 将输入特征拉直
        tf.keras.layers.Dense(128, activation='relu'),  # 此层神经元数量为经验值
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(
    optimizer='adam',  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数（已经过概率化分布）
    metrics=['sparse_categorical_accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
