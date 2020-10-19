import tensorflow as tf
import numpy as np

# 数据标注
print('---------------')
features = tf.constant([12, 20, 10, 23])
labels = tf.constant([0, 1, 0, 1])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)

for element in dataset:
    print(element)

# 求张量梯度
with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w, 2)

grad = tape.gradient(loss, w)
print(grad)

# 枚举
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)

# 独热码
print('-----------')
classes = 3
labels = tf.constant([1, 0, 2])
output = tf.one_hot(labels, depth=classes)
print(output)

# 使输出符合概率分布
y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
print(y_pro)

print('------------')

# 自更新
w = tf.Variable(4)
print(w)
w.assign_sub(1)
print(w)
