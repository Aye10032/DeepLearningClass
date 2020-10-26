import tensorflow as tf
import numpy as np
from PIL import Image
import os

train_path = '../data/mnist_image_label/mnist_train_jpg_60000/'
train_txt = '../data/mnist_image_label/mnist_train_jpg_60000.txt'
test_path = '../data/mnist_image_label/mnist_test_jpg_10000/'
test_txt = '../data/mnist_image_label/mnist_test_jpg_10000.txt'

x_train_save_path = '../data/mnist_image_label/mnist_x_train.npy'
y_train_save_path = '../data/mnist_image_label/mnist_y_train.npy'
x_test_save_path = '../data/mnist_image_label/mnist_x_test.npy'
y_test_save_path = '../data/mnist_image_label/mnist_y_test.npy'


def generate(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y_ = [], []
    for content in contents:
        value = content.split()
        img_path = path + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img / 255.
        x.append(img)
        y_.append(value[1])
        print('loading: ' + content)

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_


if os.path.exists(x_train_save_path) \
        and os.path.exists(y_train_save_path) \
        and os.path.exists(x_test_save_path) \
        and os.path.exists(y_test_save_path):
    print('------------load datasets-------------')
    x_train_save = np.load(x_train_save_path)
    y_train = np.load(y_train_save_path)
    x_test_save = np.load(x_test_save_path)
    y_test = np.load(y_test_save_path)

    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:
    print('------------generate datasets-------------')
    x_train, y_train = generate(train_path, train_txt)
    x_test, y_test = generate(test_path, test_txt)

    print('------------save datasets-------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_save_path, x_train_save)
    np.save(y_train_save_path, y_train)
    np.save(x_test_save_path, x_test_save)
    np.save(y_test_save_path, y_test)

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
