from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.externals import joblib

# Set random seed
np.random.seed(0)

rfc = joblib.load('model.pkl') 






score = rfc.score(X_test, Y_test)

print (score)