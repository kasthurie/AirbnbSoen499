# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics

# load data
data = pd.read_csv('../data/AB_NYC_2019.csv')

# data preparation
def clean_data(data: pd.DataFrame):
    data = data.drop(columns=['id', 'name', 'host_id', 'host_name'])
    data['last_review'] = pd.to_datetime(data['last_review'], infer_datetime_format=True)

    earliest_dt = min(data['last_review'])
    df = data.fillna({'reviews_per_month': 0, 'last_review': earliest_dt})

    df['last_review'] = df['last_review'].apply(lambda dt: dt.toordinal()-earliest_dt.toordinal())

    # one-hot encode categorical data
    data = pd.get_dummies(df)

    return data

new_data = clean_data(data)
print(new_data.isnull().sum())

# prediction using linearRegression model
x = new_data.drop(['price'], axis=1)
y = new_data['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=353)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluated Metrics
# MAE(Mean Absolute Error) shows the difference between predictions and actual values
MAE = mean_absolute_error(y_test, y_pred)
print('MAE(Mean Absolute Error) = ', MAE)
# MSE (Mean Squared Error)
MSE = mean_squared_error(y_test, y_pred)
print('MSE(mean squared error) = ', MSE)
# RMSE(Root mean squared error) shows how accurately the model predicts the response
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE(Root mean squared error) = ', RMSE)
# R^2(R2 score) will be calculated to find the goodness of fit measure
accuracy = r2_score(y_test, y_pred)
print('accuracy(R2 score) = ', accuracy)

# Evaluated predictions
plt.figure(figsize=(15, 7))
plt.xlim(-10, 50)
sb.regplot(y=y_test, x=y_pred, color='red')
plt.title('Evaluated predictions', fontsize=15)
plt.xlabel('Predictions')
plt.ylabel('Test')
plt.show()
