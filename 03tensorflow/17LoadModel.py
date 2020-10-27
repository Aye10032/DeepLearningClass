import tensorflow as tf
import numpy as np
from PIL import Image

check_point_path = './checkpoint/mnist.ckpt'
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(check_point_path)

img = Image.open('../data/test2.png')
img = img.resize((28, 28), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))
img_arr = 255 - img_arr
img_arr = img_arr / 255.0
x_predict = img_arr[tf.newaxis, ...]
result = model.predict(x_predict)
pred = tf.argmax(result, axis=1)
print('\n')
tf.print(pred)
