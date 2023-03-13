import numpy as np

array = np.array([[1, 2, 3],
                  [2, 3, 4]])

print(array)
print('number of dim:', array.ndim)
print('shape:', array.shape)
print('size:', array.size)

print('----------------------------------')

array = np.array([2, 23, 4], dtype=np.int64)
print(array.dtype)
array = np.array([2, 23, 4], dtype=np.float32)
print(array.dtype)
print('----------------------------------')

array = np.zeros((3, 4), dtype=int)
print(array)
print('----------------------------------')

array = np.arange(10, 20, 2)
print(array)
array = np.arange(12).reshape((3, 4))
print(array)
print('----------------------------------')

array = np.linspace(1, 10, 5)
print(array)
array = np.linspace(1, 10, 9).reshape((3, 3))
print(array)
