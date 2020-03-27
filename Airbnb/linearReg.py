# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load data
data = pd.read_csv('../data/AB_NYC_2019.csv')

# data preparation
data.drop(['last_review', 'id', 'name', 'host_id', 'host_name'], axis=1, inplace=True)
data.fillna({'reviews_per_month': 0}, inplace=True)

# model price distribution on whole data set
sb.distplot(data['price'], rug=True)

print(data.isnull().sum())
print(data['room_type'].value_counts())




