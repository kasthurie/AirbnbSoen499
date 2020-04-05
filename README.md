## AirbnbSoen499

Analysis of a dataset to recommend in which neighbourhood in New York one should invest for a successful Airbnb business. 

## Team Members:
1. Kasthurie Paramasivampillai- 40025088
2. Harraj Sandhu- 40081706
3. Gleb Pobudzey - 40062046
4. Niloofar kaypour- 40092034

# Proposal

## Abstract
Airbnb is an online rental marketplace that gained a huge popularity over the years. In this paper, we will try to predict the price of Airbnb listings in New York city. We will describe this idea and its aspects in more detail such as how different neighborhoods, availability and other features can have an impact on the pricing of different types of rental properties in Airbnb. A dataset from Kaggle will be used for the data analysis. Then, we will implement regression and random forest algorithms and discuss the results. 

## 1. Introduction
<b> 1.1 Context</b><br/>

Airbnb is an online rental marketplace that allows hosts to rent out accommodations such as rooms and houses. This rental marketplace started in 2007 and gained a huge popularity among people. Today, Airbnb contains around 3 million properties to rent out around the world, roughly in 65 thousand of cities. A lot of travelers are preferring Airbnb over hotels and motels for various reasons. The cost and the diversity of accommodations, from single room to lighthouses are the main reasons for its success. However, renters often have difficulty in determining their rental properties’ prices.

<b>1.2 Objectives <br/></b>

The goal of this project is to analyze the dataset and predict the price of the different types of accommodations in the neighborhoods of New York. The project will be to analyze a dataset using two techniques seen in class. We will use clustering and random forest to classify data and make predictions.

<b>1.3 Problem to Solve <br/></b>

One of the challenges that most hosts face, is to determine the rental price per night of their accommodation. Airbnb allows the hosts to use the Airbnb website and check at other similar listings through filtering by price, number of rooms, etc., Although this method can give the renters a little insight on the price ranges, they can still be confused since the price has to get updated often, depending on the market prices at that time.

<b>1.4 Related Work<br/></b>

There is a similar study called Predicting Airbnb prices with machine learning and location data made in 2019.  The study was made to predict the [prices of Airbnb listings of the City of Edinburgh, Scotland](https://bit.ly/2OQp7Wg). This research proves that a number of factors such as the availability, number of reviews, and type of property are the most important features that determine the price of a rental property. 

## 2. Materials and Methods
<b>2.1 Dataset(s)<br/></b>

For the analysis, the dataset called New York City Airbnb Open Data from [Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) will be used. This dataset is for public use and is taken from the website called [Inside Airbnb](http://insideairbnb.com/). This dataset contains information and metrics for Airbnb listings in New York City for 2019. It contains 16 columns including ID (listing ID), Name (name of the listing), Host ID, Host Name, Neighbourhood Group(location), Neighbourhood (area), Latitude, Longitude, Room type, Price, Minimum Night, Number of review, Last review, review per month, Calculated host listing count, Availability 365. This dataset has enough information to make predictions and draw conclusions on the pricing of the listings. This dataset will also be used to determine the features that contribute the most to the prediction of the price. For instance, the columns last review and room type are expected to contribute more to the price than columns such as hostname.

<b>2.1 Clustering Technique<br/></b>

First, we will use supervised learning, particularly the regression algorithm to predict the pricing based on features such as types of accommodations, neighborhood, etc. We will use the multiple linear regression since our dataset has more than two variables. This algorithm gives us a better picture of which feature impacts the most the output (the price in our case) and how different features are related to each other. In order to do so, we would first need to separate our dataset into attributes (X variable) and labels (pricing), which is the Y variable. Then we will split 80% of the data to the training set, 20% of the data to test and will train the model. Finally, we can compare the actual and predicted results.

<b>2.3 Random Forest<br/></b>

The second technique that we will use is random forest. Random forest is a combination of many decision trees used for feature ranking. Decision trees build regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. We will use this technique for classification of data and will allow us to predict which features have more weight and contribute the most on the pricing of the listings. Therefore, we will use random forest techniques which are used for both classification and regression. It is an ensemble of randomized decision trees. For regression it calculates the average of decision trees as the target value. So, in our data set we will use  it for identifying which feature plays an important role in the pricing of Airbnb. For example, if a particular Airbnb has a hot tub facility and if we remove that then how it’s going to affect the price of that Airbnb will be analyzed. We will also try to identify the number of factors which affect the price of Airbnb and by using random forest we can perform this task by checking the importance of particular features offered by Airbnb and how it affects the price.
