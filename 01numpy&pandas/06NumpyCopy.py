import numpy as np

a = np.arange(4)
print(a)

b = a
c = a
d = b

a[0] = 114
print(a)
print(d)
print(b is a)

d[1:3] = [1919, 810]
print(a)

f = a.copy()
a[3] = 514
print(a)
print(f)
