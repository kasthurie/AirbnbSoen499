# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# load data
data = pd.read_csv('../data/AB_NYC_2019.csv')

# data preparation
data.drop(['last_review', 'id', 'name', 'host_id', 'host_name'], axis=1, inplace=True)
data.fillna({'reviews_per_month': 0}, inplace=True)

print(data.isnull().sum())
Exploratory Data Analysis
sb.distplot(data['price'], rug=True)
# data.boxplot(column='price', by='neighbourhood_group')
# data.boxplot(column='price', by='neighbourhood')
# data.boxplot(column='price', by='room_type')
# data.boxplot(column='price', by='number_of_reviews')
# plt.figure(figsize=(20, 8))
# plt.scatter(np.log(1+data['number_of_reviews']), data['price'])
# plt.title('Price vs log(number_of_reviews)')
# data.plot.scatter(x='longitude', y='latitude', c='price', cmap='cool', alpha=0.5)
# data[data['price'] < data['price'].mean()].plot.scatter(x='longitude', y='latitude', c='price', cmap='cool', alpha=0.5)
# data[data['price'] > data['price'].mean()].plot.scatter(x='longitude', y='latitude', c='price', cmap='cool', alpha=0.5)
# plt.show()

# feature engineering
# data['log_reviews'] = np.log(1+data['number_of_reviews'])
# data = pd.get_dummies(data)
#
# # prepare linearRegression model
# x = data.copy().drop('price', axis=1)
# y = data['price'].copy()
# x_train, x_test, y_train, y_test = train_test_split(x, y)
#
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# y_pred = lr.predict(x_test)
#
# r2_score = lr.score((y_test, y_pred))
# print(r2_score)
#



