import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# Load Data
dataframe = pandas.read_csv("MSFT.csv", delimiter=",", usecols=(1,2,3))
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:2].astype(float)
Y = dataset[:,2]

X_train, X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
Y_train, Y_test = Y[:int(Y.shape[0]*0.80)],Y[int(Y.shape[0]*0.80):]


model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          epochs=20,
          batch_size=128)
score, acc = model.evaluate(X_test, Y_test, batch_size=128)

print(score)
print(acc)


'''
# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''


'''
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
'''