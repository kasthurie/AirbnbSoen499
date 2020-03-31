# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#read data from csv file
data = pd.read_csv('../data/AB_NYC_2019.csv')
