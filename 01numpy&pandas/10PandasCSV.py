import pandas as pd

data = pd.read_csv('../data/students.csv')

print(data.shape)

data.to_pickle('../data/students.pickle')
