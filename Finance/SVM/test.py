#To help us perform math operations
import numpy as np
np.set_printoptions(suppress=True)
#to plot our data and model visually
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Step 1 - Define our data


#Input data - Of the form [X value, Y value, Bias term]
X = np.array([
    [36529, 112.62, -1],
    [36530, 113.81, -1],
    [36531, 110, -1],
    [36532, 111.44, -1],
    [36535, 112.25, -1],
])

y = np.array([-1,1,-1,1,1])

#dataset = np.loadtxt("MSFT.csv", skiprows=1, delimiter=",", usecols=(1,2,3))

#X = dataset[:,0:3]
#y = dataset[:,3]


#X = X[0:10].astype(float)
#y = y[0:10].astype(float)

scl = MinMaxScaler()
X = scl.fit_transform(X)



#lets plot these examples on a 2D graph!
#for each example
for d, sample in enumerate(X):
    # Plot the negative samples (the first 2)
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples (the last 3)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Print a possible hyperplane, that is seperating the two classes.
#we'll two points and draw the line between them (naive guess)
plt.plot([-2,6],[6,0.5])