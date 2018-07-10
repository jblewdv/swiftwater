from keras.models import Sequential
from keras.layers import Dense

# Create model
model = Sequential()
# Add layers
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
# Compile
# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# Generate Numpy arrays (bring in stock data)
# ...


# Train the model with .fit()
model.fit(x, y, epochs=0, batch_size=0)
# ^^^ fill with correct data using docs

# I'm pretty sure there's more...

