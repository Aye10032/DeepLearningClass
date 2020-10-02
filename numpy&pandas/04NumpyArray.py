import numpy as np

a = np.array([1, 1, 1])
b = np.array([2, 2, 2])

print(a)
print(b)

print('----------------------------------')

c = np.vstack((a, b))
print(c, c.shape)

d = np.hstack((a, b))
print(d)

a = a[:, np.newaxis]
b = b[:, np.newaxis]
print(np.hstack((a, b)))

print('----------------------------------')

c = np.concatenate((a, b, b, a), axis=0)
print(c)
c = np.concatenate((a, b, b, a), axis=1)
print(c)
