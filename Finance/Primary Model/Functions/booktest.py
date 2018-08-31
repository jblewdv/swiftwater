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

# INITIALIZE RANDOM SEED AND SCALER
np.random.seed(0)
scl = MinMaxScaler()

# LOAD DATA
data = pd.read_csv('../CSV_Data/Raw_Data_Labels/Delta14.csv', engine='python', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13), skipfooter=14)
 

# SET FEATURES(x) AND LABELS(y) DATASETS
X = data.iloc[:,0:12]
Y = data.iloc[:,12]

# SCALE FEATURES(x)
X = scl.fit_transform(X)

# SPLIT DATA INTO TRAIN AND TEST SEGMENTS
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=None, test_size=0.2)


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)

# Just gives Accuracy
xval = cross_val_score(sgd, X_train, Y_train, cv=3, scoring="accuracy")
print("Accuracy:", xval)
 
# Gives Confusion Matrix
Y_train_preds = cross_val_predict(sgd, X_train, Y_train, cv=3)
conmax = confusion_matrix(Y_train, Y_train_preds)
#print(conmax)

# Gives precision and recall scores
prec_score = precision_score(Y_train, Y_train_preds)
rec_score = recall_score(Y_train, Y_train_preds)
#print("Precision:", prec_score)
#print("Recall:", rec_score)

# F1 Score
f1 = f1_score(Y_train, Y_train_preds)
#print("F1 Score:", f1)



Y_scores = cross_val_predict(sgd, X_train, Y_train, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


precisions, recalls, thresholds = precision_recall_curve(Y_train, Y_scores)

def plot_vs(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0,1])

#plot_vs(precisions, recalls, thresholds)
#plt.show()


# ROC CURVE
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(Y_train, Y_scores)

def plot_roc(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

#plot_roc(fpr, tpr)
#plt.show()

roc_score = roc_auc_score(Y_train, Y_scores)
#print(roc_score)


# RANDOM FOREST STUFF
rfc = RandomForestClassifier()
Y_proba_forest = cross_val_predict(rfc, X_train, Y_train, cv=3, method="predict_proba")

Y_scores_forest = Y_proba_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(Y_train, Y_scores_forest)

'''
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()
'''

#print(roc_auc_score(Y_train, Y_scores_forest))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=400,
    algorithm='SAMME.R', learning_rate=0.5)

ada_clf.fit(X_train, Y_train)
print(ada_clf.score(X_test, Y_test))



