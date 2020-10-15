import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 50)
y = 0.1 * x

plt.figure()
plt.plot(x, y, linewidth=10, color='b', zorder=1)
plt.ylim(-2, 2)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
    label.set_zorder(2)

plt.show()
