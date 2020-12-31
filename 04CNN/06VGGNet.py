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
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')

        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()
        self.a5 = Activation('relu')

        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()
        self.a6 = Activation('relu')

        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p7 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d7 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()
        self.a8 = Activation('relu')

        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()
        self.a9 = Activation('relu')

        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p10 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d10 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()
        self.a11 = Activation('relu')

        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()
        self.a12 = Activation('relu')

        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p13 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d13 = Dropout(0.2)

        self.flatten = Flatten()
        self.fa = Dense(512, activation=tf.keras.activations.relu)
        self.da = Dropout(0.2)
        self.fb = Dense(512, activation=tf.keras.activations.relu)
        self.db = Dropout(0.2)
        self.fc = Dense(10, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.b1(inputs)
        inputs = self.a1(inputs)

        inputs = self.c2(inputs)
        inputs = self.b2(inputs)
        inputs = self.a2(inputs)
        inputs = self.p2(inputs)
        inputs = self.d2(inputs)

        inputs = self.c3(inputs)
        inputs = self.b3(inputs)
        inputs = self.a3(inputs)

        inputs = self.c4(inputs)
        inputs = self.b4(inputs)
        inputs = self.a4(inputs)
        inputs = self.p4(inputs)
        inputs = self.d4(inputs)

        inputs = self.c5(inputs)
        inputs = self.b5(inputs)
        inputs = self.a5(inputs)

        inputs = self.c6(inputs)
        inputs = self.b6(inputs)
        inputs = self.a6(inputs)

        inputs = self.c7(inputs)
        inputs = self.b7(inputs)
        inputs = self.a7(inputs)
        inputs = self.p7(inputs)
        inputs = self.d7(inputs)

        inputs = self.c8(inputs)
        inputs = self.b8(inputs)
        inputs = self.a8(inputs)

        inputs = self.c9(inputs)
        inputs = self.b9(inputs)
        inputs = self.a9(inputs)

        inputs = self.c10(inputs)
        inputs = self.b10(inputs)
        inputs = self.a10(inputs)
        inputs = self.p10(inputs)
        inputs = self.d10(inputs)

        inputs = self.c11(inputs)
        inputs = self.b11(inputs)
        inputs = self.a11(inputs)

        inputs = self.c12(inputs)
        inputs = self.b12(inputs)
        inputs = self.a12(inputs)

        inputs = self.c13(inputs)
        inputs = self.b13(inputs)
        inputs = self.a13(inputs)
        inputs = self.p13(inputs)
        inputs = self.d13(inputs)

        inputs = self.flatten(inputs)
        inputs = self.fa(inputs)
        inputs = self.da(inputs)
        inputs = self.fb(inputs)
        inputs = self.db(inputs)
        outputs = self.fc(inputs)

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
