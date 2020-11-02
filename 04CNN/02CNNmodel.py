import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        # 卷积（特征提取）
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # C:卷积层--6个卷积核，每个尺寸为5*5，全零填充
        self.b1 = BatchNormalization()  # B:BN层--批标准化
        self.a1 = Activation('relu')  # A:激活层--激活
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # P: 池化层--最大池化，尺寸2*2，步长2
        self.d1 = Dropout(0.2)  # D:dropout层--20%的神经元休眠

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.b1(inputs)
        inputs = self.a1(inputs)
        inputs = self.p1(inputs)
        inputs = self.d1(inputs)

        inputs = self.flatten(inputs)
        inputs = self.f1(inputs)
        inputs = self.d2(inputs)
        outputs = self.f2(inputs)

        return outputs


model = BaseModel()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

check_point_path = './checkpoint/mnist.ckpt'
if os.path.exists(check_point_path + '.index'):
    print('-----------load model------------')
    model.load_weights(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=10,
                    validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')

file.close()

# 绘制ACC和LOSS曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('ACC')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('loss')
plt.legend()
plt.show()
