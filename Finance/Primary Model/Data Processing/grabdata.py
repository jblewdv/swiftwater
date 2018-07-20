# Grab data function for Swiftwater Investments algorithm series
# Copyright 2018 Swiftwater, Joshua Blew
####################################################

# <-- IMPORTS -->
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import numpy as np
import pandas as pd

#########################################

# <-- GLOBAL VARS -->
#input_ticker = input("Grab Full Data for ticker symbol... ")
input_ticker = 'MMM'
ALPHAVANTAGE_API_KEY = 'T1CY4ZZ01MNPN4YF'

#########################################

# <-- DATAFRAME & CSV CREATION -->
ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas', indexing_type='date')
ti = TechIndicators(key=ALPHAVANTAGE_API_KEY, output_format='pandas')

# Price & Indicator Data
data, meta_data = ts.get_daily(symbol=input_ticker, outputsize='full')
macd, macd_meta = ti.get_macd(symbol=input_ticker, interval='daily', series_type='close')
rsi, rsi_meta = ti.get_rsi(symbol=input_ticker, interval='daily', time_period=14, series_type='close')
willr, willr_meta = ti.get_willr(symbol=input_ticker, interval='daily', time_period=14)
adx, adx_meta = ti.get_adx(symbol=input_ticker, interval='daily', time_period=14)
mom, mom_meta = ti.get_mom(symbol=input_ticker, interval='daily', time_period=10, series_type='close')

all_vals = [data, macd, rsi, willr, adx, mom]

final_df = pd.concat(all_vals, axis=1, sort=True) # Sort arg may need to be False, leaving it blank raises Pandas error
final_df = final_df.dropna()
df = final_df.iloc[::-1]

#########################################
'''
# Init Pandas df
labels = []

for index, row in df.iterrows():
    
    # Vars
    close_price = row['4. close']
    open_price = row['1. open']

    if close_price > open_price:
    	labels.append("1")
    else:
    	labels.append("0")

labels_df = pd.DataFrame(np.array(labels), columns=['Label'])
print (labels_df)
'''
# Need to concat this Lables df with the prior df to then export to csv

df.to_csv('output_dataMMM.csv')



















