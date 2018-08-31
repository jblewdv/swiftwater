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
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import pprint

# *************************************************************************
# *************************************************************************

# INITIALIZE RANDOM SEED AND SCALER
seed = 0
np.random.seed(seed)
scaler = MinMaxScaler()

# LOAD DATA
data = pd.read_csv('AAPL.csv', engine='python', usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), skipfooter=5000)

# x and y for prediction
X = data.iloc[:,0:13]
Y = data.iloc[:,13]

# data for analysis
fullData = data.iloc[:,0:15]

# SCALE FEATURES(x)
X[['1. open', '2. high', '3. low', '4. close', '5. adjusted close', '8. split coefficient', 'MACD', 'MACD_Hist', 'MACD_Signal', 'RSI', 'WILLR', 'ADX', 'MOM']] = scaler.fit_transform(X[['1. open', '2. high', '3. low', '4. close', '5. adjusted close', '8. split coefficient', 'MACD', 'MACD_Hist', 'MACD_Signal', 'RSI', 'WILLR', 'ADX', 'MOM']])
# *************************************************************************

# TRAIN: 50%, VALIDATE: 25%. TEST: 25%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=None, test_size=0.50, shuffle=True, random_state=seed)

fullDataTrain, fullDataTest = train_test_split(fullData, train_size=None, test_size=0.50, shuffle=True, random_state=seed)

X_test, X_validate = X_test[:int(X_train.shape[0]*0.50)], X_test[int(X_train.shape[0]*0.50):]
Y_test, Y_validate = Y_test[:int(Y_train.shape[0]*0.50)], Y_test[int(Y_train.shape[0]*0.50):]

fullDataTest, fullDatValidate = fullDataTest[:int(fullDataTest.shape[0]*0.50)], fullDataTest[int(fullDataTest.shape[0]*0.50):]
# *************************************************************************

# converts pandas df to list of tuples
fullDataTestTuples = list(fullDataTest.itertuples(index=False, name=None))

# converts these tuples into a list of lists
fullDataTestList = []
for i in fullDataTestTuples:
    fullDataTestList.append(list(i))

# grabs shuffled index list 
myIndex = list(fullDataTest.index.values)
# *************************************************************************

# INIT RFC MODEL
rfc = RandomForestClassifier(n_estimators=70)

# FIT MODEL 2 TRAIN DATA
rfc.fit(X_train, Y_train)

score = rfc.score(X_test, Y_test)
probs = rfc.predict_proba(X_test).tolist()
preds = rfc.predict(X_test)
cm = pd.crosstab(Y_test, preds, rownames=['True'], colnames=['Predicted'], margins=True)


print ("Old Accuracy = %s perc." % round((score*100), 4))
print (cm)

'''
# *************************************************************************
for i, x, y, z in zip(myIndex, fullDataTestList, preds, probs):
    x.insert(0, (i))
    x.append(y)
    
    zeroProb, oneProb = round((z[0]), 4), round((z[1]), 4)
    x.append(zeroProb)
    x.append(oneProb)

fullDataTestList = sorted(fullDataTestList, key=lambda x: x[0])


# *************************************************************************
# *************************************************************************

threshold = 0.75
min_val = (abs(1-(threshold-.01))/(threshold-.01))
max_val = ((threshold-.01)/(abs(1-(threshold-.01))))

certain, uncertain = [],[]
certain_correct, certain_incorrect, uncertain_correct, uncertain_incorrect = 0,0,0,0
fullPreds = []


for actual, predicted, prob in zip(Y_test, preds, probs):

    zero_prob = prob[0]
    one_prob = prob[1]

    instance = [actual, predicted, zero_prob, one_prob]
    fullPreds.append(instance)

    # SPLIT INTO CERTAIN & UNCERTAIN
    if instance[2] >= threshold or instance[3] >= threshold:
        certain.append(instance)

        if instance[0] != instance[1]:
            certain_incorrect +=1
        else:
            certain_correct +=1

    else:
        # ADD UNCERTAIN WEIGHT
        init_weight = instance[2]/instance[3]
        scaled_weight = round((init_weight-min_val)/(max_val-min_val), 4)
        instance.append(scaled_weight)
        uncertain.append(instance)

        if instance[0] != instance[1]:
            uncertain_incorrect +=1
        else:
            uncertain_correct +=1

# *************************************************************************
# *************************************************************************

# GLOBAL VARIABLES
badCounts, badIndexes, combined = [],[],[]
count = 0


# RECORDS INDEX OF EACH INCORRECT PREDICTION
for i in fullPreds:
    if i[0] != i[1]:
        badCounts.append(count)
    else:
        pass
    count +=1

# RECORDS RESPECTIVE CSV ROW INDEX OF INCORRECT PREDICTION
for i in badCounts:
    badIndexes.append(myIndex[i])

# CREATES NEW LIST OF INDEX AND IT'S RESPECTIVE CSV ROW
for i, x in zip(badCounts, badIndexes):
    combined.append([i,x+2])

# ADDS ROW INDEX TO CORRESPONDING PD DF ROW
for i in combined:
    fullDataTestList[i[0]].insert(0, (i[1]))

completedIncorrect = []
for i in fullDataTestList:
    if len(i) == 13:
        completedIncorrect.append(i)


# *************************************************************************
# *************************************************************************

lookBack = 5
nums = list(np.arange(1,lookBack+1))

# Lookback Function
def getLookBack(index, steps):
    from statistics import mean
    results = []

    for i in steps:
        for x in fullDataTestList:
            if x[0] == index-i:
                results.append(x)
    
    correctCount, zeroCount, oneCount = 0,0,0
    zeroMean, oneMean = [],[]

    for i in results:
        if i[13] == i[14]:
            correctCount +=1
        if i[14] == 0:
            zeroCount -=1
        if i[14] == 1:
            oneCount +=1
        zeroMean.append(i[15])
        oneMean.append(i[16])
        
    accuracy = correctCount/len(results)
    # negative bias = more 0's, positive bias = more 1's
    bias = zeroCount+oneCount
    zeroMean = mean(zeroMean)
    oneMean = mean(oneMean)
    output = [accuracy, bias, zeroMean, oneMean]

    return output

#output = getLookBack(35, nums)


# *************************************************************************

# 
def checkFallout(lookback, fallout):
    maxPrice = max(fullData['4. close'][:lookback])
    currentPrice = fullData.iloc[0]['4. close']
    
    if currentPrice <= maxPrice*(1-fallout):
        # need to sell!
        return True
    else:
        # everything's okay :)
        return False



# *************************************************************************
# *************************************************************************

#newAcc = round(((certain_correct/(len(X_test)-len(uncertain)))*100), 4)
#totalPreds = (len(X_test)-len(uncertain))

print ("There are %s certain predictions" % len(certain))
print ("And there are %s uncertain predictions" % len(uncertain))
print ("Out of %s testing samples" % len(X_test))
print("\n")

print ("UNCERTAIN predictions... \n%s are correct and %s are incorrect" % (uncertain_correct, uncertain_incorrect))
print ("CERTAIN predictions... \n%s are correct and %s are incorrect" % (certain_correct, certain_incorrect))
print("\n")

print ("New Adjusted Accuracy = %s" % newAcc)
print ("With %s total predictions" % (len(X_test)-len(uncertain)))
print ("Which is %s predictions/month" % (totalPreds/55))'''

# *************************************************************************
# *************************************************************************

# new dataframe col for multiclass classification of price diffs
binLabels = []

for index, row in fullData.iterrows():
    currentDiff = abs(row[14])
    threshold = row[4] * 0.01
    if 0 <= currentDiff < threshold:
        binLabels.append(0)
    else:
        binLabels.append(1)
        
binLabelsPD = pd.Series(binLabels)
fullData['Price Diff Labels'] = binLabelsPD.values

#fullData.to_csv('dothisagain.csv')

