#To help us perform math operations
import numpy as np
np.set_printoptions(suppress=True)
#to plot our data and model visually
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Step 1 - Define our data


#Input data - Of the form [X value, Y value, Bias term]
#Input data - Of the form [X value, Y value, Bias term]
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

#Associated output labels - First 2 examples are labeled '-1' and last 3 are labeled '+1'
y = np.array([-1,-1,1,1,1])
#dataset = np.loadtxt("MSFT.csv", skiprows=1, delimiter=",", usecols=(1,2,3))

#X = dataset[:,0:3]
#y = dataset[:,3]


#X = X[0:10].astype(float)
#y = y[0:10].astype(float)

scl = MinMaxScaler()
#X = scl.fit_transform(X)



#lets perform stochastic gradient descent to learn the seperating hyperplane between both classes

def svm_sgd_plot(X, Y):
    #Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0]))
    #The learning rate
    eta = 1
    #how many iterations to train for
    epochs = 100000
    #store misclassifications so we can plot how they change over time
    errors = []

    #training part, gradient descent part
    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            #misclassification
            if (Y[i]*np.dot(X[i], w)) < 1:
                #misclassified update for ours weights
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
                #correct classification, update our weights
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)
    
    return w

for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add our test samples
#plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
#plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

w = svm_sgd_plot(X,y)
print(w)
#they decrease over time! Our SVM is learning the optimal hyperplane

# Print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')
plt.show()

