import tensorflow as tf
import numpy as np

a = tf.constant([1, 5], dtype=tf.int32)
print(a)
print(a.dtype)
print(a.shape)

b = np.arange(0, 5)
print(b)

c = tf.convert_to_tensor(b, dtype=tf.int32)
print(c)
print(c.shape)
print(c.dtype)

d = tf.zeros(3)
e = tf.ones(3)
f = tf.fill([3, 3], 114)

print(d)
print(e)
print(f)

# 正态分布
r1 = tf.random.normal([2, 2], mean=0.2, stddev=1)  # mean=均值， stddev=标准差）

# 截断式正态分布
r2 = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print(r1)
print(r2)

# 随机数
r = tf.random.uniform([2, 2], minval=0, maxval=1)
print(r)
