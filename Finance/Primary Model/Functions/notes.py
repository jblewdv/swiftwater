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

'''
# SAVE MODEL TO DISK
from sklearn.externals import joblib
joblib.dump(rfc, 'model.pkl')
# LOADING
rfc = joblib.load('model.pkl') 

from sklearn.metrics import mean_squared_error
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# PUSH PREDICTED LABELS TO CSV
df = pd.DataFrame(np.array(final_predictions).reshape(len(final_predictions), 1), columns = ['Labels'])
df.to_csv('testPredictions.csv')


# FIND BEST HYPER-PARAMS
max_feature_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
max_depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
n_estimator_range = [10, 100, 300, 500, 1000]

# BEST HYPER-PARAMETERS *need to test again with sample_leaf parameter stuff
# 0.8809106830122592
# {'n_estimators': 1000, 'max_features': 9, 'max_depth': 10}

param_grid = dict(n_estimators=n_estimator_range, max_features=max_feature_range, max_depth=max_depth_range)

from sklearn.model_selection import RandomizedSearchCV

rand = RandomizedSearchCV(rfc, param_grid, cv=5, scoring='accuracy')
rand.fit(X_train, Y_train)

print (rand.best_score_)
print (rand.best_params_)


# *************************************************************************

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

# INITIALIZE RANDOM SEED AND SCALER
np.random.seed(0)
scl = MinMaxScaler()

# LOAD DATA
data = pd.read_csv('../CSV_Data/Raw_Data_Labels/Delta14.csv', engine='python', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13), skipfooter=14)

# SET FEATURES(x) AND LABELS(y) DATASETS
fvecs = data.iloc[:,0:12]
labels = data.iloc[:,12]

# SCALE FEATURES(x)
fvecs = scl.fit_transform(fvecs)

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(fvecs, labels)

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#print(grid_search.best_params_)





from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest_reg, fvecs, labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Dev:", scores.std())

display_scores(forest_rmse_scores)


# *************************************************************************
'''
'''# PUSH PREDICTED LABELS TO CSV
df = pd.DataFrame(np.array(completedIncorrect).reshape(len(completedIncorrect), 13), columns = ['index', '1. open', '2. high', '3. low', '4. close', '5. volume', 'MACD', 'MACD_Hist', 'MACD_Signal', 'RSI', 'WILLR', 'ADX', 'MOM'])
df.to_csv('badOnes.csv')'''


# PUSH PREDICTED LABELS TO CSV
#df = pd.DataFrame(np.array(preds).reshape(len(preds), 1), columns = ['Labels'])
#f.to_csv('newPhase.csv')



# og bin system
'''import math
# dynamically creating bins
max_diff = max(fullData['Price Diffs'])
min_diff = min(fullData['Price Diffs'])

binThreshold = 5
bins = []

skipThreshold = 1

def roundup(x):
    return int(math.ceil(x / binThreshold)) * binThreshold

upperBound = roundup(max_diff)
lowerBound = -(roundup(abs(min_diff)))

bins = list(np.arange(lowerBound, upperBound+1, binThreshold))
bins.append(skipThreshold)
bins.append(-skipThreshold)
bins.sort()

# adds diffs to correct bin
PriceDiffs = {}
for index, row in fullData.iterrows():
    for binIndex, x in enumerate(bins):
        currentDiff = row[14]
        if x <= currentDiff < bins[binIndex+1]:
            key = x
            PriceDiffs.setdefault(key, [])
            val = [index, currentDiff]
            PriceDiffs[key].append(val)

classifiers = list(np.arange(0, len(bins)))

# new dataframe col for multiclass classification of price diffs
binLabels = []

#Get a list of keys from dictionary which has the given value
def getKey(dict, value):
    for i in dict.items():
        for x in i[1]:
            if x[1] == value:
                return i[0]
            else:
                pass

for index, row in fullData.iterrows():
    givenBin = getKey(PriceDiffs, row["Price Diffs"])
    for i, x in zip(bins, classifiers):
        if i == givenBin:
            binLabels.append(x)
        else:
            pass

binLabelsPD = pd.Series(binLabels)
fullData['Price Diff Labels'] = binLabelsPD.values

fullData.to_csv('priceDiffMultiClass.csv')
'''

'''else:
        if x <= currentDiff:
                key = x
                PriceDiffs.setdefault(key, [])
                val = [index, currentDiff]
                PriceDiffs[key].append(val)
        
        
#Get a list of keys from dictionary which has the given value
def getKey(dict, value):
    for i in dict.items():
        for x in i[1]:
            if x[1] == value:
                return i[0]

for index, row in fullData.iterrows():
    givenBin = getKey(PriceDiffs, abs(row["Price Diffs"]))
    for i, x in zip(bins, classifiers):
        if i == givenBin:
            binLabels.append(x)'''
