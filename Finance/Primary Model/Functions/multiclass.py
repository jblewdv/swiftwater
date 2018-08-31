
# IMPORTS
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pprint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# *************************************************************************
# *************************************************************************

# INITIALIZE RANDOM SEED AND SCALER
seed = 0
np.random.seed(seed)
scaler = MinMaxScaler()

# LOAD DATA
data = pd.read_csv('dothisagain.csv', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,16))

# x and y for prediction
X = data.iloc[:,0:13]
Y = data.iloc[:,13]

# data for analysis
#fullData = data.iloc[:,0:15]

# SCALE FEATURES(x)
X[['1. open', '2. high', '3. low', '4. close', '5. adjusted close', '8. split coefficient', 'MACD', 'MACD_Hist', 'MACD_Signal', 'RSI', 'WILLR', 'ADX', 'MOM']] = scaler.fit_transform(X[['1. open', '2. high', '3. low', '4. close', '5. adjusted close', '8. split coefficient', 'MACD', 'MACD_Hist', 'MACD_Signal', 'RSI', 'WILLR', 'ADX', 'MOM']])

'''# dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=None, test_size=0.50, shuffle=True, random_state=seed)

# *************************************************************************



# random forest
rfc2 = RandomForestClassifier(n_estimators=100)

rfc2.fit(X_train, Y_train)

score = rfc2.score(X_test, Y_test)
probs = rfc2.predict_proba(X_test).tolist()
preds = rfc2.predict(X_test)
cm = pd.crosstab(Y_test, preds, rownames=['True'], colnames=['Predicted'], margins=True)


print ("Accuracy = %s perc." % round((score*100), 4))
print (cm)
'''
threshold = 0.7

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
        uncertain.append(instance)
        if instance[0] != instance[1]:
            uncertain_incorrect +=1
        else:
            uncertain_correct +=1

newAcc = round(((certain_correct/(len(X_test)-len(uncertain)))*100), 4)
totalPreds = (len(X_test)-len(uncertain))

print ("There are %s certain predictions" % len(certain))
print ("And there are %s uncertain predictions" % len(uncertain))
print ("Out of %s testing samples" % len(X_test))
print("\n")

print ("UNCERTAIN predictions... \n%s are correct and %s are incorrect" % (uncertain_correct, uncertain_incorrect))
print ("CERTAIN predictions... \n%s are correct and %s are incorrect" % (certain_correct, certain_incorrect))
print("\n")

print ("New Adjusted Accuracy = %s" % newAcc)
print ("With %s total predictions" % (len(X_test)-len(uncertain)))
#print ("Which is %s predictions/month" % (totalPreds/55))'''