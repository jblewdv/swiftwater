import pandas as pd
import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

lr = LogisticRegression()
svc = LinearSVC()
rfc = RandomForestClassifier(n_estimators=100)


pima = pd.read_csv('pima-indians-diabetes.csv')

from sklearn.model_selection import train_test_split

train, test = train_test_split(pima, test_size=0.2)

train_feat = train.iloc[:,:8]
train_targ = train.iloc[:,8]

test_feat = test.iloc[:,:8]
test_targ = test.iloc[:,8]

from sklearn.metrics import accuracy_score

'''
# Logistic Regression
lr.fit(train_feat, train_targ)
train_score = lr.score(train_feat, train_targ)
test_score = lr.score(test_feat, test_targ)

print('Accuracy = ', accuracy_score(test_feat, test_targ))
'''

# Random Forest Classifier
rfc.fit(train_feat, train_targ)
score = rfc.score(test_feat, test_targ)
print(score)
preds = rfc.predict(test_feat)

# Create confusion matrix
print pd.crosstab(test_targ, preds, rownames=['Actual'], colnames=['Predicted'])

'''
# SVC
svc.fit(train_feat, train_targ)
score2 = svc.score(test_feat, test_targ)
print(score2)
'''
