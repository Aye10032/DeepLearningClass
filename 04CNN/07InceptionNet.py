import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class ConvBNRelu(Model):
    def __init__(self, ch, kernel_size=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.modle = tf.keras.models.Sequential([
            Conv2D(ch, kernel_size, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, inputs, **kwargs):
        outputs = self.modle(inputs, training=False)

        return outputs


class InceptionBlock(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlock, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernel_size=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernel_size=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernel_size=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernel_size=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernel_size=5, strides=1)
        self.P4_1 = MaxPool2D(2, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernel_size=1, strides=strides)

    def call(self, inputs, training=None, mask=None):
        inputs1 = self.c1(inputs)
        inputs2_1 = self.c2_1(inputs)
        inputs2_2 = self.c2_2(inputs2_1)
        inputs3_1 = self.c3_1(inputs)
        inputs3_2 = self.c3_2(inputs3_1)
        inputs4_1 = self.P4_1(inputs)
        inputs4_2 = self.c4_2(inputs4_1)

        outputs = tf.concat([inputs1, inputs2_2, inputs3_2, inputs4_2], axis=3)  # 沿深度方向堆叠

        return outputs


class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlock(self.out_channels, strides=2)
                else:
                    block = InceptionBlock(self.out_channels, strides=1)
                self.blocks.add(block)

            self.out_channels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, inputs, **kwargs):
        inputs = self.c1(inputs)
        inputs = self.blocks(inputs)
        inputs = self.p1(inputs)
        outputs = self.f1(inputs)

        return outputs


model = Inception10(num_blocks=2, num_classes=10)

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
