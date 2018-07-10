# Takes stock ticker input and spits out CSV with last 100 days price info
# Copyright 2018 Joshua Blew
################################################

from alpha_vantage.timeseries import TimeSeries
import numpy as np
from sklearn.preprocessing import scale
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# Make plots bigger
# matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)


# Connect to Alpha Vantage
ticker = input("Stock To Use: ")

ts = TimeSeries(key=['T1CY4ZZ01MNPN4YF'], output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')





# CSV file creation
fh = open("%s.csv" % ticker, 'w+')

for i,date in enumerate(data.index):
	fh.write("%s,%.2f\n" % (date, data['4. close'][i]))
fh.close()

pth = '%s.csv' % ticker
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(0, 1))

print(A)



'''
data['4. close'].plot()
plt.title('Daily Times Series for the ' + ticker + ' stock (1 day)')
plt.grid()
plt.show()
'''
