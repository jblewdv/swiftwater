import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# Load Data
data = pd.read_csv('BIGDATATEST.csv', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
data.dropna(how="all", inplace=True) 

fvecs = data.iloc[:,0:12]
labels = data.iloc[:,12]


scl = MinMaxScaler()
fvecs = scl.fit_transform(fvecs)

X_train, X_test = fvecs[:int(fvecs.shape[0]*0.8)],fvecs[int(fvecs.shape[0]*0.8):]
Y_train, Y_test = labels[:int(labels.shape[0]*0.8)],labels[int(labels.shape[0]*0.8):]


model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
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