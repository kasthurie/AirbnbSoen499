## AirbnbSoen499

Analysis of a dataset to predict price of Airbnb in different neighbourhood in New York.

## Team Members:
1. Kasthurie Paramasivampillai- 40025088
2. Harraj Sandhu- 40081706
3. Gleb Pobudzey - 40062046
4. Niloofar kaypour- 40092034

## Abstract
Airbnb is an online rental marketplace that gained a huge popularity over the years. In this paper, we will predict the price of Airbnb listings in New York city. First, we will describe this idea and its aspects in more detail such as how different neighborhoods, availability, room types and other features can have an impact on the pricing of different types of rental properties in Airbnb. Second, we will describe our dataset and analyze its features. Then, we will implement linear regression and random forest, which are two algorithms that we decided to use to predict the price. Finally, we will check the performance of our models from each of these algorithms and will compare the results. 

## 1. Introduction
<b> 1.1 Context</b><br/>

Airbnb is an online rental marketplace that allows hosts to rent out accommodations such as rooms and houses. This rental marketplace started in 2007 and gained a huge popularity among people. Today, Airbnb contains around 3 million properties to rent out around the world, roughly in 65 thousand cities. A lot of travelers are preferring Airbnb over hotels and motels for various reasons. The cost and the diversity of accommodations, from single room to lighthouses are the main reasons for its success. However, renters often have difficulty in determining their rental properties’ prices.

<b>1.2 Objectives <br/></b>

The goal of this project is to analyze the dataset and predict the price of the different types of accommodations in the neighborhoods of New York. The project will be to analyze a dataset using two techniques seen in class. We will use linear regression and random forest, which are both supervised machine learning regression algorithms, to predict the price.

<b>1.3 Problem to Solve <br/></b>

One of the challenges that most hosts face, is to determine the rental price per night of their accommodation. Airbnb allows the hosts to use the Airbnb website and check at other similar listings through filtering by price, number of rooms, etc., Although this method can give the renters a little insight on the price ranges, they can still be confused since the price has to get updated often, depending on the market prices at that time.

<b>1.4 Related Work<br/></b>

There is a similar study called Predicting Airbnb prices with machine learning and location data made in 2019.  The study was made to predict the [prices of Airbnb listings of the City of Edinburgh, Scotland](https://bit.ly/2OQp7Wg). This research proves that a number of factors are more important than other features to determine the price of a rental property. The two models that were used are Spatial Hedonic Price Model (OLS Regression), with the LinearRegression and the Gradient Boosting method, with the XGBRegressor. They used MSE and R squared to check the accuracy of the models. They concluded that the top five features that are the most important in predicting the price are the following:<br>

-If the rental is the entire flat or not (room_type_Entire home/apt)<br>
-How many people the property accommodates (accommodates)<br>
-The type of property (property_type_Other)<br>
-The number of bathrooms (bathrooms)<br>
-How many days are available to book out of the next 90 (availability_90)<br>

They also got a R squared of 0.51 and a RMSE of 0.26 for the spatial hedonic model, so we can say that the features explain approximately 51% of the variance in the target variable and the RMSE can be interpreted as the standard deviation of the unexplained variation. The R squared for the XGboost model was 0.65, which is slightly better and a smaller value of RMSE (~0.18). 


## 2. Materials and Methods
<b>2.1 Dataset(s)<br/></b>

For the analysis, the dataset called “AB_NYC_2019” from [Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) was used. This dataset is for public use and is taken from the website called [Inside Airbnb](http://insideairbnb.com/). This dataset contains information and metrics for Airbnb listings in New York City for 2019. It has about 49,000 rows and the following columns:<br>

-ID (listing ID)<br>
-Name (name of the listing)<br>
-Host ID<br>
-Host Name<br>
-Neighbourhood Group(location)<br>
-Neighbourhood (area)<br>
-Latitude<br>
-Longitude<br>
-Room type<br>
-Price<br>
-Minimum Night<br>
-Number of reviews<br>
-Last review<br>
-review per month<br>
-Calculated host listing count<br>
-Availability 365<br>

This dataset has enough information to make predictions and draw conclusions on the pricing of the listings. This dataset was also used to determine the features that contribute the most to the prediction of the price. For instance, the columns last review and room type are expected to contribute more to the price than columns such as hostname.

<b>2.2 Data Pre-Processing<br/></b>

First, we cleaned and organized the dataset. This step will eliminate data that are incomplete and thus will allow us to get more accurate and reliable results. We dropped the rows with too many missing values. We also replaced missing data with statistical estimation of the missing values (mean, median, mode). We decided to handle the missing data in the “Last Review” column by replacing it by the earliest year and the NULL values in “Reviews per Month” by replacing it by 0. We dropped columns that are irrelevant to predict the price. These columns are Listing ID, Name of listing, Host ID, and Host Name. We also used categorized features to prepare our data for prediction. At the end of this step, the data was ready to be processed. Figure 1 shows a snippet of how we used dataframe to clean data.

![](https://imgur.com/efHBRv7.png)

The following columns are our finalized features that we used for the implementation of algorithms:<br>

-Neighbourhood Group(location)<br>
-Neighbourhood (area)<br>
-Latitude<br>
-Longitude<br>
-Room type<br>
-Price<br>
-Minimum Night<br>
-Number of reviews<br>
-Last review<br>
-Review per month<br>
-Calculated host listing count<br>
-Availability 365<br>

<b>2.3 Data Analysis<br/></b>

The next step was to better understand the dataset and see the relationship between different features and the price. Figure 2 shows the distribution of price between different neighbourhood groups.

![](https://imgur.com/KVWG00v.png)

We can observe that Manhattan has the most expensive Airbnb in New York and the second highest is Brooklyn. The cheapest Airbnb can be found in Staten island.

Figure 3 shows a graph of longitude versus latitude. Each data point in the graph represents the price of an airbnb listing. We can observe that the majority of airbnb in New York are categorised as either very cheap or cheap. 

![](https://imgur.com/l3FdZ0C.png)

Furthermore, Figure 4 is a graph of price versus the log of reviews and Table 1 shows the mean price of room type in the neighbourhoods.

![](https://imgur.com/4fVIj1B.png)
![](https://imgur.com/3Km1iP4.png)

From the correlation matrix shown in Figure 5, we can observe that the features which have the most impact on the price are longitude, latitude and the number of reviews. 

![](https://imgur.com/Gog3Buo.png)

## 3. Results
<b>3.1 Technique 1: Multiple Linear Regression<br/></b>

Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). Therefore, the dependent variable (y) can be calculated from a linear combination of the input variables (x). 

We used supervised learning, particularly the regression algorithm to predict the pricing based on features such as types of accommodations, neighborhood, etc. We used the multiple linear regression since our dataset has more than two variables. This algorithm gives us a better picture of which feature impacts the most the output (the price in our case) and how different features are related to each other. We separated our dataset into attributes (x) and labels (pricing), which is the y variable. Then we split 90% of the data to be the training set, 10% of the data to be the testing set and trained our model. Finally, we compared the actual and predicted results.

By implementing linear regression on our dataset, we were able to see the extent to which there is a linear relationship between the price and the other features. To evaluate the accuracy of our model and to compare the results later with Random Forest, we used mean squared error and R squared as metrics and did the prediction on both training and test data to ensure we do not have an overfitting problem. Table 2 below summarizes the results obtained from linear regression.

![](https://imgur.com/O64IY9D.png)

<b>3.2 Technique 2: Supervised learning with decision trees<br/></b>

<b>3.2.1 Single decision tree<br/></b>

Decision trees are simple to interpret, to understand and can also be combined with other decision techniques. Furthermore, decision trees use a white box model, meaning any output derived by the model can be explained by a tree traversal.

The training error was nearly 0, which means the decision tree model overfit the training data. Even though the model predicted the training very well, the model was useless since it did not generalize to new data.
 
<b>3.2.2 Random forest<br/></b>

A decision tree is built on the entire dataset, whereas a random forest randomly selected rows and specific features to build multiple decision trees from and then averaged the results. Therefore, we can expect an ensemble of trees to generalize better. Indeed, the training error went up and the test error went down.

<b>3.2.3 Random forest w/ grid search<br/></b>

Decision trees have many hyperparameters that influence its construction, namely the maximum depth, number of estimators (number of trees in a forest), the minimum number of leafs splits per node). Finding the optimal hyperparameters for a decision tree is an NP-hard problem, therefore we performed a grid search to get the approximate best values. Indeed, we defined a range of values for each hyperparameter and tried all the values generated by their cartesian product.

Since it was computationally an intensive task, we only tuned two hyperparameters: the number of estimators, and the maximum depth of the decision tree. As we can see in the results in Table 3, the test error was lower with tuned hyperparameters.

Furthermore, with more processing power, we can extend the ranges of values considered and increase the number of hyperparameters. Another viable solution is to randomly search for parameters in a defined grid. Indeed, we might have wrong intuition as to the range of values of the optimal hyperparameter values and therefore a randomized search approach can remove those biases.

<b>3.2.4 Random forest w/ grid search and cross validation<br/></b>
 
Decision trees are unstable since a small change in data can drastically change the structure of the optimal decision tree. Therefore, decision trees are prone to overfitting. One solution is a K fold cross validation.
 
We split the data in K parts, holded out each split as test data and used the rest as training data. Since we rotated the test set, we reduced the model’s selection bias and it generalized better on independent data.

The results show that cross validation is very successful, we reduced the test error by a factor of 2.

![](https://imgur.com/GdFFqXw.png)

Using random forest, we were also able to determine the features that have the more weight on predicting the price. From Table 4, we can see that longitude and latitude are the most important.

![](https://imgur.com/bVQ1teX.png)

## 4. Algorithms Comparison

Using both random forest and the correlation matrix, we were able to see that the most important features that determine the price of airbnb are the longitude and latitude. This can be explained since the location is very important when choosing an Airbnb. For example, listings that are located near tourist attractions can be more expensive than others. 

As we mentioned above, we used R2 score and MSE to evaluate the accuracy of our models.  R2 score is the proportion of variance in the dependent variable that is predictable from the independent variable(s). Our linear regression model shows a low level of correlation between the two variables with a R2 score of 0.16. For our random forest model, the values of R2 score are all negative, suggesting that our model is performing poorly. Adding XGBoost could have given us better results. In a [similar analysis of Airbnb price](https://www.kaggle.com/jrw2200/smart-pricing-with-xgb-rfr-interpretations?fbclid=IwAR2v2VgTZVXx6htcGX8nXWr0SezHgBlOdlyUZp2jh7cVqfV27xfnJxICCYc), using XGBoost improved the performance of the model and gave a R2 score of 0.601344.

Mean square error (MSE) is the average of the square of the errors. The larger the number the larger the error. From Table 3, we can observe that the MSE improved a lot in Random Forest with Grid Search and Cross Validation. The value went from 82596.29 to 25803.340. For our linear regression model, we got a MSE of 28404.77, which is higher than what we got for Random Forest (with Grid Search & CV). 

## 5. Discussion

Since we noticed that reviews per month and number of reviews have less importance in predicting price (Table 4) , we can drop these columns and run linear regression again to see if there are any improvements in the performance. For future work, we can tune more hyperparameters when doing random forest with grid search and see if the results get better. This was not possible in the limited time we had since the process is computationally intensive. As we mentioned earlier, we can also implement XGBoost and see if there is an improvement in the results.
