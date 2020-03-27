import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('../data/AB_NYC_2019.csv')
df.drop(['last_review'], axis=1, inplace=True)
df.drop(['id'], axis=1, inplace=True)
df.drop(['name'], axis=1, inplace=True)
df.drop(['host_id'], axis=1, inplace=True)
df.drop(['host_name'], axis=1, inplace=True)
df.fillna({'reviews_per_month': 0}, inplace=True)

print(df.isnull().sum())
print(df.shape)
print(df)





