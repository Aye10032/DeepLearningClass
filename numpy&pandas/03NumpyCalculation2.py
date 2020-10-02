import numpy as np

A = np.arange(2, 14).reshape((3, 4))
print(A)

print(np.argmin(A))
print(np.argmax(A))

print(np.mean(A))
print(A.mean())
print(np.average(A))

print(np.cumsum(A))
print(np.diff(A))

print(np.nonzero(A))

A = np.arange(14, 2, -1).reshape((3, 4))
print(np.sort(A))

print(np.transpose(A))
print(A.T)

print(np.clip(A, 5, 9))

print('----------------------------------')

A = np.arange(3, 15).reshape((3, 4))

print(A)
print(A[2])
print(A[2][1])
print(A[2, :])
print(A[:, 1])
print(A[1, 1:3])

print('----------------------------------')

for row in A:
    print(row)

for col in A.T:
    print(col)

print(A.flatten())
for item in A.flat:
    print(item)
