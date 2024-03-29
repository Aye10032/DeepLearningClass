import numpy as np
import pandas as pd

dates = pd.date_range('20201001', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

print(df)
print('-----------------------')

print(df['A'])
print(df[0:3])
print('-----------------------')

print(df.loc['20201003'])
print(df.loc[:, ['A', 'C']])
print('-----------------------')

print(df.iloc[3])
print(df.iloc[3:5, 1:3])
print('-----------------------')

print(df[df.A > 8])
print('-----------------------')

df.iloc[2, 2] = 1111
print(df)
df.A[df.A > 4] = 0
print(df)

df['E'] = np.nan
df['F'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20201001', periods=6))
print(df)
