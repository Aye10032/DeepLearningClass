import pandas as pd
import numpy as np

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)

res = pd.merge(left, right, on='key')
print(res)
print('-------------------------')

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)

res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
print(res)

res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
print(res)

res = pd.merge(left, right, on=['key1', 'key2'], how='left')
print(res)
print('-------------------------')

df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})
print(df1)
print(df2)

res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
print(res)

res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
print(res)
print('-------------------------')

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C1', 'C2'],
                      'D': ['D0', 'D1', 'D2']},
                     index=['K0', 'K2', 'K3'])
print(left)
print(right)

res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
print(res)
print('-------------------------')

df1 = pd.DataFrame({'k': [0, 1, 2], 'age': [1, 2, 3]})
df2 = pd.DataFrame({'k': [0, 0, 3], 'age': [4, 5, 6]})
print(df1)
print(df2)

res = pd.merge(df1, df2, on='k', suffixes=['_1', '_2'], how='inner')
print(res)
