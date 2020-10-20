from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 数据集读入

x_data = load_iris().data
y_data = load_iris().target

# 数据集乱序

np.random.seed(116)  # 统一的随机种子，生成同样的打乱
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 截取训练集与测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 配对特征与标签，打包为小搓
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)  # 32对为一组
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 可训练参数
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))  # 4个特征、3个类别--4个输入节点、3个输出神经元
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))  # 与W的类别维度一致

lr = 0.1  # 学习率
train_loss_results = []  # 每轮的loss
test_acc = []  # 每轮的acc
epoch = 500  # 循环次数
loss_all = 0  # 纪录每轮4个step所生成的4个loss的和

# 嵌套循环迭代，更新参数结构，显示当前loss
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))  # 均方差·
            loss_all += loss.numpy()

        grads = tape.gradient(loss, [w1, b1])  # 损失函数对W1、B1求偏导

        # 梯度更新 w1 = w1 - lr * w1_grad     b1 = b1 - lr * b1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    print('Epoch {}, loss: {}'.format(epoch, loss_all / 4))
    train_loss_results.append(loss_all / 4)
    loss_all = 0

    # 测试
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)

        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct / total_number
    test_acc.append(acc)
    print('Test acc: ', acc)
    print('--------------------')

plt.figure()
plt.title('loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label='$Loss$')
plt.legend()

plt.figure()
plt.title('acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='$Acc$')
plt.legend(

)

plt.show()
