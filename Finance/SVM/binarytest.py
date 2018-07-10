from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
np.set_printoptions(suppress=True)
# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
dataset = np.loadtxt("AAPL-3.csv", delimiter=",", skiprows=2, usecols=(4,7))

# split into input (X) and output (Y) variables
X = dataset[:,0:1]
Y = dataset[:,1]

#X = scale(X)

X_train, X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
Y_train, Y_test = Y[:int(Y.shape[0]*0.80)],Y[int(Y.shape[0]*0.80):]

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

'''
# define and fit the final model
model = Sequential()
model.add(Dense(8, input_dim=1, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=500, batch_size=64, verbose=0)
predictions = model.predict_classes(X_test)

for i in range(len(X_test)):
	print("X=%s, Predicted=%s, Actual=%s" % (X_test[i], predictions[i], Y_test[i]))

# evaluate the model
scores = model.evaluate(X_test, Y_test, batch_size=64)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
'''



print ('Creating model...')
model = Sequential()
model.add(LSTM(32, activation='sigmoid', recurrent_activation='hard_sigmoid', return_sequences=True))
model.add(LSTM(32, activation='sigmoid', recurrent_activation='hard_sigmoid'))
model.add(Dense(1, activation='sigmoid'))

print ('Compiling...')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print ("Fitting model...")
hist = model.fit(X_train, Y_train, batch_size=64, epochs=50, verbose = 1)

predictions = model.predict_classes(X_test)

for i in range(len(X_test)):
	print("X=%s, Predicted=%s, Actual=%s" % (X_test[i], predictions[i], Y_test[i]))

# evaluate the model
scores = model.evaluate(X_test, Y_test, batch_size=64)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



