import matplotlib.pyplot as plt
import numpy as np

a = np.array([0.313, 0.365, 0.423,
              0.365, 0.439, 0.525,
              0.423, 0.525, 0.651]).reshape(3, 3)

plt.imshow(a, interpolation='nearest', cmap=plt.cm.bone, origin='lower')
plt.colorbar(shrink=0.9)

plt.xticks(())
plt.yticks(())
plt.show()
