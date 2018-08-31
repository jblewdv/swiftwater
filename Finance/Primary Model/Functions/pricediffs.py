# *************************************************************************
# 
# SWIFTWATER INVESTMENTS
# __________________
# 
#  Copyright (c) 2018 Joshua Blew
#  All Rights Reserved.
# 
# NOTICE:  All information contained herein is, and remains
# the property of Swiftwater Investments and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Swiftwater Investments
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Swiftwater Investments.
# /

# IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# *************************************************************************
# *************************************************************************

# INITIALIZE RANDOM SEED AND SCALER
seed = 0
np.random.seed(seed)
scaler = MinMaxScaler()

# LOAD DATA (Rows 13,14,15,16 are Diffs)
data = pd.read_csv('AAPL-PriceDiffs.csv', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,14))

# SET FEATURES(x) AND DIFFS(y) DATASETS
X = data.iloc[:,0:12]
Y = data.iloc[:,12]

# SCALE FEATURES(x)
X[['1. open', '2. high', '3. low', '4. close', '5. volume', 'MACD', 'MACD_Hist', 'MACD_Signal', 'RSI', 'WILLR', 'ADX', 'MOM']] = scaler.fit_transform(X[['1. open', '2. high', '3. low', '4. close', '5. volume', 'MACD', 'MACD_Hist', 'MACD_Signal', 'RSI', 'WILLR', 'ADX', 'MOM']])

# TRAIN: 50%, VALIDATE: 25%. TEST: 25%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=None, test_size=0.50, shuffle=False, random_state=seed)
X_test, X_validate = X_test[:int(X_train.shape[0]*0.50)], X_test[int(X_train.shape[0]*0.50):]
Y_test, Y_validate = Y_test[:int(Y_train.shape[0]*0.50)], Y_test[int(Y_train.shape[0]*0.50):]

# *************************************************************************
# *************************************************************************
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
# REGRESSION MODELING

# decision tree regression
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, Y_train)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# random forest regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, Y_train)

# ridge - play around with alpha
ridge_reg = Ridge(alpha=1, solver='cholesky')
ridge_reg.fit(X_train, Y_train)

# lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, Y_train)




# decision tree regression
#predictions = tree_reg.predict(X_validate)
#tree_mse = mean_squared_error(Y_validate, predictions)
#tree_rmse = np.sqrt(tree_mse)
#print (tree_mse, tree_rmse)

# linear regression
#predictions = lin_reg.predict(X_validate)
#lin_mse = mean_squared_error(Y_validate, predictions)
#lin_rmse = np.sqrt(lin_mse)
#print (lin_mse, lin_rmse)
# predicting on training data yields 2.4ish rmse
# while on validating data it yields 40.35

# random forest ...^^
#predictions = forest_reg.predict(X_validate)
#forest_mse = mean_squared_error(Y_validate, predictions)
#forest_rmse = np.sqrt(forest_mse)
#print (forest_mse, forest_rmse)

# ridge
#predictions = ridge_reg.predict(X_validate)
#ridge_mse = mean_squared_error(Y_validate, predictions)
#ridge_rmse = np.sqrt(ridge_mse)
#print (ridge_mse, ridge_rmse)

# lasso
predictions = lasso_reg.predict(X_validate)
lasso_mse = mean_squared_error(Y_validate, predictions)
lasso_rmse = np.sqrt(lasso_mse)
print (lasso_mse, lasso_rmse)


# cross validation
# decision tree
#scores = cross_val_score(tree_reg, X_validate, Y_validate, scoring='neg_mean_squared_error', cv=10)
#tree_rmse_scores = np.sqrt(-scores)

# linear regression
#scores = cross_val_score(lin_reg, X_validate, Y_validate, scoring='neg_mean_squared_error', cv=10)
#lin_rmse_scores = np.sqrt(-scores)

# random forest
#scores = cross_val_score(forest_reg, X_validate, Y_validate, scoring='neg_mean_squared_error', cv=10)
#forest_rmse_scores = np.sqrt(-scores)

# ridge
#scores = cross_val_score(ridge_reg, X_validate, Y_validate, scoring='neg_mean_squared_error', cv=10)
#ridge_rmse_scores = np.sqrt(-scores)

# lassp
scores = cross_val_score(lasso_reg, X_validate, Y_validate, scoring='neg_mean_squared_error', cv=10)
lasso_rmse_scores = np.sqrt(-scores)


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("STD:", scores.std())

#print_scores(lin_rmse_scores)
#print_scores(tree_rmse_scores)
#print_scores(forest_rmse_scores)
#print_scores(ridge_rmse_scores)
print_scores(lasso_rmse_scores)