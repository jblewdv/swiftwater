from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Set random seed
np.random.seed(0)

# Load Data
data = pd.read_csv('2weeksApart.csv', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
data.dropna(how="all", inplace=True) 

fvecs = data.iloc[:,0:12]
labels = data.iloc[:,12]
print (len(fvecs))

scl = MinMaxScaler()
fvecs = scl.fit_transform(fvecs)

X_train, X_test = fvecs[:int(fvecs.shape[0]*0.8)],fvecs[int(fvecs.shape[0]*0.8):]
Y_train, Y_test = labels[:int(labels.shape[0]*0.8)],labels[int(labels.shape[0]*0.8):]

rfc = RandomForestClassifier(n_estimators=400)
# Current Settings == 77.5609756097561% Accuracy
# from apple data csv (max feat =2, n estims =400, minsampleaf =2)


rfc.fit(X_train, Y_train)

preds = rfc.predict(X_test)
probs = rfc.predict_proba(X_test)
details = pd.crosstab(Y_test, preds, rownames=['Actual'], colnames=['Predicted'])
feat_imp = list(zip(X_train, rfc.feature_importances_))
score = rfc.score(X_test, Y_test)

print (score)
print (details)
print(probs[-10:])




##########################

# <-- SAVE MODEL TO DISK -->
from sklearn.externals import joblib
joblib.dump(rfc, 'model.pkl')

##########################
'''
# <-- RESULTS TAB -->

csv = monthApart.csv
Calculating the labels based on close values spread by exactly 1 month yields an 88.7% acc with 400 trees.

	   STATS
0.8873239436619719
Predicted  0.0  1.0
Actual             
0.0        355   54
1.0         50  464

[[0.135  0.865 ]
 [0.245  0.755 ]
 [0.2125 0.7875]
 [0.29   0.71  ]
 [0.3325 0.6675]
 [0.44   0.56  ]
 [0.4    0.6   ]
 [0.6525 0.3475]
 [0.7225 0.2775]
 [0.4825 0.5175]]

# ------------

csv = 2weeksApart.csv
2 week window instead of 1 month

	  STATS
0.9891891891891892
Predicted    0    1
Actual             
0          417    0
1           10  498

[[0.98   0.02  ]
 [0.9875 0.0125]
 [0.9925 0.0075]
 [0.995  0.005 ]
 [0.985  0.015 ]
 [0.995  0.005 ]
 [0.985  0.015 ]
 [0.935  0.065 ]
 [0.9525 0.0475]
 [0.9375 0.0625]]



'''
##########################