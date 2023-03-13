import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

tbCallBack = TensorBoard(log_dir='./logs')  # log 目录

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 为数据集增加一个维度，使之与网络结构匹配

image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 将图像归一化为0-1
    rotation_range=45,  # 随机45°旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=0.5  # 随机缩放阈值量
)
image_gen_train.fit(x_train)

# model = keras.models.Sequential([
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])

model = keras.applications.DenseNet169(
    include_top=True,
    weights=None,
    input_shape=(28, 28, 3),
    pooling='max',
    classes=2)

model.compile(
    optimizer='adam',  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数（已经过概率化分布）
    metrics=['sparse_categorical_accuracy']
)

model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test),
          validation_freq=1, callbacks=[tbCallBack])
model.summary()
