import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 50)
y = 2 * x + 1

plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = 2 * x0 + 1
plt.scatter(x0, y0, s=50)  # 散点图
plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)  # 黑色虚线

plt.annotate(r'$(1,3)$', xy=(x0, y0), xycoords='data', xytext=(+30, -30), textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.text(2, 3.5, r'$y = 2x + 1$', fontdict={'size': 16, 'color': 'r'})

plt.show()
