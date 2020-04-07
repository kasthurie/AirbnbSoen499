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

# load data
data = pd.read_csv('../data/AB_NYC_2019.csv')

# data preparation
data.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1, inplace=True)
data.fillna({'reviews_per_month': 0}, inplace=True)

Min_Price = min(data['price'].value_counts())
Max_Price = max(data['price'].value_counts())
Mean_Price = data['price'].value_counts().mean()

data['Category'] = data['price'].apply(lambda p: 'Super Expensive' if p > Mean_Price * 4
else ('Expensive' if Mean_Price * 2 <= p < Mean_Price * 4
      else ('Medium' if Mean_Price <= p < Mean_Price * 2
            else ('Reasonable' if Mean_Price * (3 / 4) <= p < Mean_Price
                  else ('Cheap' if Mean_Price / 2 <= p < Mean_Price * (3 / 4)
                        else 'very cheap')))))

# print(data['Category'].value_counts())

data.drop(['neighbourhood', 'number_of_reviews', 'reviews_per_month'], axis=1, inplace=True)

# encode data
def Encode(f):
    for column in data.columns[data.columns.isin(['neighbourhood_group', 'room_type', 'Category'])]:
        data[column] = data[column].factorize()[0]
    return data

new_data = Encode(data.copy())
print(new_data.isnull().sum())

# prediction using linearRegression model
x = new_data.iloc[:, [0, 1, 2, 3, 7, 8]]
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
