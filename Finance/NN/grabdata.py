# Grab data function for Swiftwater Investments algorithm series
# Copyright 2018 Swiftwater, Joshua Blew
####################################################

# Imports
from alpha_vantage.timeseries import timeseries



# Variables to get:
# Price Data
# RSI
# MACD
# WillR %
# MOM
# ADX

input_ticker = input("Grab Full Data for ticker symbol... ")

# <-- Data Query Strings -->
MACD = 'https://www.alphavantage.co/query?function=MACD&symbol=%s&interval=daily&series_type=open&apikey=demo' % input_ticker
RSI = 'https://www.alphavantage.co/query?function=RSI&symbol=MSFT&interval=15min&time_period=14&series_type=close&apikey=demo'
WillR = 'https://www.alphavantage.co/query?function=WILLR&symbol=MSFT&interval=15min&time_period=10&apikey=demo'
ADX = 'https://www.alphavantage.co/query?function=ADX&symbol=MSFT&interval=15min&time_period=10&apikey=demo'
MOM = 'https://www.alphavantage.co/query?function=MOM&symbol=MSFT&interval=15min&time_period=10&series_type=close&apikey=demo'






print (MACD)









'''


ts = TimeSeries(key=['T1CY4ZZ01MNPN4YF'], output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')







# CSV Creation
fh = open("%s.csv" % input_ticker, 'w+')

for i,date in enumerate(data.index):
	fh.write("%s,%.2f\n" % (date, data['4. close'][i]))
fh.close()

pth = '%s.csv' % ticker
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(0, 1))

print(A)

'''

