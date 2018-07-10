from keras.models import Sequential
from keras.layers import Dense
import numpy

# load pima indians dataset
dataset = numpy.loadtxt("MSFT.csv", delimiter=",", usecols=(1,2))
# split into input (X) and output (Y) variables
dates = dataset[:,0:1]
prices = dataset[:,1]
# create model
model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(dates, prices, epochs=25, batch_size=10)
# evaluate the model
scores = model.evaluate(dates, prices)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))