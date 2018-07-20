import numpy as np
from keras.models import Sequential	
from keras.layers import Dense, LSTM
	
from keras.utils import np_utils

from matplotlib import pyplot as plt
from sklearn.preprocessing import scale


# Step 1: Load Data
dataset = np.loadtxt('MSFT.csv', delimiter=',', usecols=(1,2))

# Step 2: Preprocessing The Data
dataset = scale(dataset)
dates = dataset[:,0:1].reshape(-1,1)
prices = dataset[:,1].reshape(-1,1)

X_train, X_test = dates[:int(dates.shape[0]*0.80)],dates[int(dates.shape[0]*0.80):]
Y_train, Y_test = prices[:int(prices.shape[0]*0.80)],prices[int(prices.shape[0]*0.80):]

# Step 3: Building The Model
model = Sequential()
model.add(LSTM(256, input_shape=(1,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

print(score)



############