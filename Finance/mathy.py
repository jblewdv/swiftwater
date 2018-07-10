# Mathematical operations on selected stocks
# Copyright 2018 Swiftwater, Joshua Blew
####################################################

# <-- IMPORTS -->
from alpha_vantage.timeseries import TimeSeries
import numpy as np
import pandas as pd

#########################################

# <-- GLOBAL VARS -->
#input_ticker = input("Grab Full Data for ticker symbol... ")
input_ticker = 'AAPL'
ALPHAVANTAGE_API_KEY = 'T1CY4ZZ01MNPN4YF'

#########################################

# <-- DATAFRAME & CSV CREATION -->
ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas', indexing_type='date')

# Price & Indicator Data
data, meta_data = ts.get_daily(symbol=input_ticker, outputsize='full')

#########################################

# <-- PROCESS DATA -->
# Drop all except Close
df = data.drop(['1. open', '2. high', '3. low', '5. volume'], axis=1)

# Sort by price
df = df.sort_values(by=['4. close'])

dates = list(df.index.values)
diffs = []

for index, i in enumerate(dates):
	delta = dates[index] - dates[index-1]

	diffs.append(delta)

print(diffs[0:10])









