import tensorflow as tf
import numpy as np

a = tf.random.uniform([3, 3], minval=0, maxval=10)
print(a)

# 强制转换数据类型
b = tf.cast(a, dtype=tf.float64)
print(b)

# 计算张量维度上的最小值/最大值
print(tf.reduce_min(a))
print(tf.reduce_max(a))

x = tf.constant([[1, 2, 3],
                 [2, 3, 2]], dtype=tf.int32)

# 计算平均值
print(tf.reduce_mean(x, axis=0))  # axis=0表示纵向计算
# 求和
print(tf.reduce_sum(x, axis=1))  # axis=1表示横向计算

# 将数据标记为可训练
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
print(w)

# 张量四则运算
x1 = tf.constant([[2, 3, 4],
                  [4, 5, 6]], dtype=tf.float32)
x2 = tf.constant([[3, 4, 5],
                  [5, 6, 7]], dtype=tf.float32)

print(tf.add(x1, x2))
print(tf.subtract(x1, x2))
print(tf.multiply(x1, x2))
print(tf.divide(x1, x2))
print('-------------------------')

# 平方、次方、开方
print(tf.square(x1))
print(tf.pow(x1, 3))
print(tf.sqrt(x1))

# 矩阵乘
x3 = np.transpose(x2)
print(tf.matmul(x1, x3))

# 求最值
test = np.array([[1, 2, 3],
                 [2, 3, 4],
                 [5, 4, 3],
                 [8, 7, 2]])
print(test)
print(tf.argmax(test, axis=0))  # 列
print(tf.argmax(test, axis=1))  # 行
