import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


dates, prices = [], []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0][8:]))
			prices.append(float(row[1]))
	return


def predict_prices(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1))

	svr_rbf = SVR(kernel='poly', C=1e3, degree=3)
	svr_rbf.fit(dates, prices)

	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='Predicted')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('SVR')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0]

get_data('AAPL.csv')

predicted_price = predict_prices(dates, prices, 35)

print (predicted_price)