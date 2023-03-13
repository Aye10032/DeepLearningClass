import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

plt.imshow(x_train[0])
plt.show()

print(x_train.shape)

model = keras.applications.DenseNet169()

model.compile(
    optimizer='adam',  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数（已经过概率化分布）
    metrics=['sparse_categorical_accuracy']
)
tbCallBack = TensorBoard(log_dir='./logs')  # log 目录

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test),
          validation_freq=1, callbacks=[tbCallBack])
model.summary()
