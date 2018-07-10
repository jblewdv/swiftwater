import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# Score result = 0.5868469472829397 (trial 1)
# 0.6090754425370848 (trial 2)

df = pd.read_csv('pima-indians-diabetes.csv')

'''
train, test = train_test_split(df, test_size=0.2)

X_train = train.iloc[:,:8]
Y_train = train.iloc[:,8]

X_test = test.iloc[:,:8]
Y_test = test.iloc[:,8]
'''


# split into input (X) and output (Y) variables
X = df.iloc[:,:8]
Y = df.iloc[:,8]

X_train, X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
Y_train, Y_test = Y[:int(Y.shape[0]*0.80)],Y[int(Y.shape[0]*0.80):]

# Build Model
model = Sequential()
model.add(Dense(20, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=0, epochs=1000)

predictions = model.predict_classes(X_test)
print(predictions)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print(scores)
 



