import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Load Data
dataset = np.loadtxt('MSFT.csv', delimiter=',', usecols=(1,2))

prices = dataset[:,1]

scl = MinMaxScaler()
#Scale the data
prices = prices.reshape(prices.shape[0],1)
prices = scl.fit_transform(prices)

#Create a function to process the data into 7 day look back slices
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)
X,y = processData(prices,3)
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]


#Build the model
model = Sequential()
model.add(LSTM(256,input_shape=(3,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),shuffle=False)


#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])

act = []
pred = []

for i in range(250):
	#i=249
	Xt = model.predict(X_test[i])
	#print('predicted:{0}, actual:{1}'.format(scl.inverse_transform(Xt),scl.inverse_transform(y_test[i].reshape(-1,1))))
	pred.append(scl.inverse_transform(Xt))
	act.append(scl.inverse_transform(y_test[i].reshape(-1,1)))

result_df = pd.DataFrame({'pred':list(np.reshape(pred, (-1))),'act':list(np.reshape(act, (-1)))})

# Plot results
Xt = model.predict(X_test)
# Red
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label="Actual", c='#b0403f')
plt.ylabel('Stock Price')
# Blue
plt.plot(scl.inverse_transform(Xt), label="Predicted", c='#5aa9ab')
plt.xlabel('Date Value')
plt.show()



'''
          pred    act
0    44.843975  42.74
1    44.409134  43.63
2    44.056095  44.08
3    43.953644  44.88
4    44.120625  44.38
5    44.353096  45.02
6    44.598236  46.13
7    44.966492  45.91
8    45.379570  46.49
9    45.750496  46.62
10   46.096180  46.05
11   46.315643  46.95
12   46.527637  47.44
13   46.765316  47.57
14   47.001381  47.86
15   47.270939  48.70
16   47.584518  48.68
17   47.926495  48.89
18   48.268101  48.87
19   48.520996  48.78
20   48.689938  49.61
21   48.873428  49.58
22   49.073399  49.46
23   49.218971  48.74
24   49.263607  48.22
25   49.154804  48.70
26   49.010399  47.98
27   48.833500  47.59
28   48.564495  47.47
29   48.283791  47.75
..         ...    ...
220  43.444324  41.82
221  43.524155  43.36
222  43.572712  43.50
223  43.698048  42.61
224  43.667694  43.89
225  43.659317  43.07
226  43.689369  43.29
227  43.720860  43.48
228  43.818485  43.04
229  43.809814  43.98
230  43.850567  44.30
231  44.022896  44.25
232  44.182495  43.48
233  44.295322  44.11
234  44.352737  43.90
235  44.389477  43.87
236  44.415318  43.91
237  44.393169  43.94
238  44.371849  43.29
239  44.305576  43.44
240  44.227554  44.26
241  44.200504  44.61
242  44.305645  45.57
243  44.563549  46.63
244  45.005589  46.75
245  45.537750  46.80
246  46.060608  47.45
247  46.536003  47.11
248  46.895573  47.00
249  47.123852  46.89

[250 rows x 2 columns]
'''