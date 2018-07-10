import numpy as np 
import pandas as pd 
import pprint

from keras.models import Sequential	
from keras.layers import Dense, Dropout, LeakyReLU
from sklearn.preprocessing import MinMaxScaler


# Set random seed
np.random.seed(0)

# Load Data
data = pd.read_csv('AAPL-Full-Data.csv', usecols=(4,5,6,7,8,11,12))

data.dropna(how="all", inplace=True) 

# Split into (x) and (y) data points
fvecs = data.iloc[:,0:6]
labels = data.iloc[:,6]

scl = MinMaxScaler()
fvecs = scl.fit_transform(fvecs)

# Vectorize Data
fvecs_np = np.matrix(fvecs).astype(np.float32)
labels_np = np.array(labels).astype(dtype=np.uint8)

# Create two new dataframes, one with the training rows, one with the test rows
X_train, X_test = fvecs_np[:int(fvecs_np.shape[0]*0.80)],fvecs_np[int(fvecs_np.shape[0]*0.80):]
Y_train, Y_test = labels_np[:int(labels_np.shape[0]*0.80)],labels_np[int(labels_np.shape[0]*0.80):]
 
# i need to increase training data size
###################

# Define NN Model
model = Sequential()
model.add(Dense(64, input_dim=6, kernel_initializer='normal'))
# ???
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, Y_train,
          epochs=350,
          batch_size=512)
score = model.evaluate(X_test, Y_test, batch_size=512)

print (score)


