import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3), padding='valid')
        self.b1 = BatchNormalization()
        self.a1 = Activation(tf.keras.activations.relu)
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3), padding='valid')
        self.b2 = BatchNormalization()
        self.a2 = Activation(tf.keras.activations.relu)
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same')
        self.a3 = Activation(tf.keras.activations.relu)

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same')
        self.a4 = Activation(tf.keras.activations.relu)

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.a5 = Activation(tf.keras.activations.relu)
        self.p5 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation=tf.keras.activations.relu)
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation=tf.keras.activations.relu)
        self.d2 = Dropout(0.5)
        self.f3 = Dense(10, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.b1(inputs)
        inputs = self.a1(inputs)
        inputs = self.p1(inputs)

        inputs = self.c2(inputs)
        inputs = self.b2(inputs)
        inputs = self.a2(inputs)
        inputs = self.p2(inputs)

        inputs = self.c3(inputs)
        inputs = self.a3(inputs)

        inputs = self.c4(inputs)
        inputs = self.a4(inputs)

        inputs = self.c5(inputs)
        inputs = self.a5(inputs)
        inputs = self.p5(inputs)

        inputs = self.flatten(inputs)
        inputs = self.f1(inputs)
        inputs = self.d1(inputs)
        inputs = self.f2(inputs)
        inputs = self.d2(inputs)
        outputs = self.f3(inputs)

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
