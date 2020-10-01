import numpy as np

a = np.array([10, 20, 30, 40])
b = np.arange(4)
print(a, b)

c = a - b
print(c)

c = b ** 2  # not b^2!
print(c)

c = 10 * np.sin(a)
print(c)

print(b < 3)  # [ True  True  True False]

print('----------------------------------')

a = np.array([[1, 1],
              [0, 1]])
b = np.arange(4).reshape((2, 2))
print(a)
print(b)

c = a * b
print(c)

c = np.dot(a, b)
print(c)
c = a.dot(b)
print(c)

print('----------------------------------')
a = np.random.random((3, 3))
print(a)

print(np.sum(a))
print(np.min(a))
print(np.max(a))

print(np.min(a, axis=1))
