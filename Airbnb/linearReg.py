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

# Exploratory Data Analysis
sb.distplot(data['price'], rug=True)
data.boxplot(column='price', by='neighbourhood_group')
data.boxplot(column='price', by='neighbourhood')
data.boxplot(column='price', by='room_type')
data.boxplot(column='price', by='reviews_per_month')
plt.figure(figsize=(20, 8))
plt.scatter(np.log(1+data['reviews_per_month']), data['price'])
plt.title('Price vs log(reviews_per_month)')
data.plot.scatter(x='longitude', y='latitude', c='price', cmap='cool', alpha=0.5)
data[data['price'] < data['price'].mean()].plot.scatter(x='longitude', y='latitude', c='price', cmap='cool', alpha=0.5)
data[data['price'] > data['price'].mean()].plot.scatter(x='longitude', y='latitude', c='price', cmap='cool', alpha=0.5)
plt.show()

print(data.isnull().sum())
# print(data['price'].value_counts())







