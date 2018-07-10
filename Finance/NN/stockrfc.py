from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Set random seed
np.random.seed(0)

# Load Data
data = pd.read_csv('AAPL-Full-Data.csv', usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
data.dropna(how="all", inplace=True) 

fvecs = data.iloc[:,0:11]
labels = data.iloc[:,11]

scl = MinMaxScaler()
fvecs = scl.fit_transform(fvecs)

X_train, X_test = fvecs[:int(fvecs.shape[0]*0.8)],fvecs[int(fvecs.shape[0]*0.8):]
Y_train, Y_test = labels[:int(labels.shape[0]*0.8)],labels[int(labels.shape[0]*0.8):]

clf = RandomForestClassifier(max_features=2, n_estimators=400, min_samples_leaf=2)
# Current Settings == 77.5609756097561% Accuracy

clf.fit(X_train, Y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)
details = pd.crosstab(Y_test, preds, rownames=['Actual'], colnames=['Predicted'])
feat_imp = list(zip(X_train, clf.feature_importances_))
score = clf.score(X_test, Y_test)

print score


