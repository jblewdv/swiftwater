from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Set random seed
np.random.seed(0)

data = pd.read_csv('pima-indians-diabetes.csv')

# Create a dataframe with the four feature variables
df = pd.DataFrame(data)

# split into input (X) and output (Y) variables
X = df.iloc[:,:8]
Y = df.iloc[:,8]

# Create two new dataframes, one with the training rows, one with the test rows
X_train, X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
Y_train, Y_test = Y[:int(Y.shape[0]*0.80)],Y[int(Y.shape[0]*0.80):]

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators=100)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(X_train, Y_train)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
preds = clf.predict(X_test)

# View the predicted probabilities of the first 10 observations
probs = clf.predict_proba(X_test)

# Create confusion matrix
print pd.crosstab(Y_test, preds, rownames=['Actual'], colnames=['Predicted'])

# View a list of the features and their importance scores
feat_imp = list(zip(X_train, clf.feature_importances_))



