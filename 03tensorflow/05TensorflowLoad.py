from sklearn.datasets import load_iris
import pandas as pd

# 数据集读入

x_data = load_iris().data
y_data = load_iris().target

print('x_data from datasets: \n', x_data)
print('y_data from datasets: \n', y_data)

pd.set_option('display.unicode.east_asian_width', True)

x_data = pd.DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
print('x_data add index: \n', x_data)

y_data = pd.DataFrame(y_data, columns=['类别'])
print('x_data add index: \n', y_data)
