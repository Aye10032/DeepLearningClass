import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure()
ax11 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
ax11.plot([1, 2], [1, 2])
ax11.set_title('ax1_title')

ax12 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
ax13 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)
ax14 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)
ax15 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)

plt.figure()
gs = gridspec.GridSpec(3, 3)
ax21 = plt.subplot(gs[0, :])
ax22 = plt.subplot(gs[1, :2])
ax23 = plt.subplot(gs[1:, 2])
ax24 = plt.subplot(gs[-1, 0])
ax25 = plt.subplot(gs[-1, -2])

plt.show()
