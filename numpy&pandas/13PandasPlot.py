import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.Series(np.random.randn(1000), index=np.arange(1000))

data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list('ABCD'))
# print(data.head(5))
data = data.cumsum()
data.plot()
data.plot.scatter(x='A', y='B')
plt.show()
