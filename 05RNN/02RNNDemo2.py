import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

input_word = 'abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
id_to_onehot = {0: [1., 0., 0., 0., 0.],
                1: [0., 1., 0., 0., 0.],
                2: [0., 0., 1., 0., 0.],
                3: [0., 0., 0., 1., 0.],
                4: [0., 0., 0., 0., 1.]}

x_train = [
    [id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']]],
    [id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']]],
    [id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']]],
    [id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']]],
    [id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']]]
]
y_train = [w_to_id['e'],
           w_to_id['a'],
           w_to_id['b'],
           w_to_id['c'],
           w_to_id['d']]

np.random.seed(6)
np.random.shuffle(x_train)
np.random.seed(6)
np.random.shuffle(y_train)
tf.random.set_seed(6)

x_train = np.reshape(x_train, (len(x_train), 4, 5))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    SimpleRNN(3),  # 记忆体个数为3个
    Dense(5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

check_point_path = './checkpoint/rnn_onehot_4pre1.ckpt'
if os.path.exists(check_point_path + '.index'):
    print('-----------load model------------')
    model.load_weights(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')

history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])

model.summary()

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')

file.close()

# 绘制ACC和LOSS曲线
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('ACC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('loss')
plt.legend()
plt.show()

# 预测程序
preNum = int(input('input the number of test alphabet:'))
for i in range(preNum):
    alphabet1 = input('input test alphabet:')
    alphabet = [id_to_onehot[w_to_id[a]] for a in alphabet1]

    alphabet = np.reshape(alphabet, (1, 4, 5))
    result = model.predict(alphabet)
    pred = tf.argmax(result, axis=1)  # 取预测结果最大
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])
