import numpy as np
#from TFANN import ANNR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale

pth = 'AAPL-DateVal.csv'
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 4))
A = scale(A)
#y is the dependent variable
y = A[:, 1].reshape(-1, 1)
#A contains the independent variable
A = A[:, 0].reshape(-1, 1)



#Number of neurons in the input layer
i = 1
#Number of neurons in the output layer
o = 1
#Number of neurons in the hidden layers
h = 32
#3 Fully-connected layers with tanh followed by linear output layer 
layers = [('F', h), ('AF', 'tanh'), ('F', h), ('AF', 'tanh'), ('F', h), ('AF', 'tanh'), ('F', o)]
mlpr = ANNR([i], layers, batchSize = 256, maxIter = 1000, tol = 0.2, reg = 1e-4, verbose = True)


#Fit the MLP to the data
#param A: numpy matrix where each row is a sample
#param y: numpy matrix of target values
def fit(self, A, y):
    m = len(A)
    #Start the tensorflow session and initializer
    #all variables
    self.sess = tf.Session()
    init = tf.initialize_all_variables()
    self.sess.run(init)
    #Begin training
    for i in range(self.mItr):
        #Batch mode or all at once
        if(self.batSz is None):
            self.sess.run(self.optmzr, feed_dict={self.x:A, self.y:y})
        else:
            for j in range(0, m, self.batSz):
                batA, batY = _NextBatch(A, y, j, self.batSz)
                self.sess.run(self.optmzr, feed_dict={self.x:batA, self.y:batY})
        err = np.sqrt(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y}) * 2.0 / m)
        if(self.vrbse):
            print("Iter " + str(i + 1) + ": " + str(err))
        if(err < self.tol):
            break

#Predict the output given the input (only run after calling fit)
#param A: The input values for which to predict outputs
#return: The predicted output values (one row per input sample)
def predict(self, A):

#Predicts the ouputs for input A and then computes the RMSE between
#The predicted values and the actualy values
#param A: The input values for which to predict outputs
#param y: The actual target values
#return: The RMSE
def score(self, A, y):




#Length of the hold-out period
nDays = 5
n = len(A)
#Learn the data
mlpr.fit(A[0:(n-nDays)], y[0:(n-nDays)])

#Begin prediction
yHat = mlpr.predict(A)
#Plot the results
mpl.plot(A, y, c='#b0403f')
mpl.plot(A, yHat, c='#5aa9ab')
mpl.show()



