import numpy as np
import pandas as pd

s = pd.Series([1, 3, 6, np.nan, 44, 1])
print(s)

dates = pd.date_range('20201001', periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)

df = pd.DataFrame({'A': 1.,
                   'B': pd.Timestamp('20201001'),
                   'C': pd.Series(1, index=list(range(4)), dtype=np.float32),
                   'D': np.array([3] * 4, dtype=np.int32),
                   'E': pd.Categorical(['test', 'train', 'test', 'train']),
                   'F': 'foo'})
print(df)
print(df.dtypes)
print(df.index)
print(df.columns)

print(df.values)

print(df.describe())
print(df.sort_index(axis=1, ascending=False))
print(df.sort_values(by='E'))
