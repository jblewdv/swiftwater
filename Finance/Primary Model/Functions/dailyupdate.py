# *************************************************************************
# 
# SWIFTWATER INVESTMENTS
# __________________
# 
#  Copyright (c) 2018 Joshua Blew
#  All Rights Reserved.
# 
# NOTICE:  All information contained herein is, and remains
# the property of Swiftwater Investments and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Swiftwater Investments
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Swiftwater Investments.
# /

# IMPORTS
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from time import sleep
import sqlite3 

# GLOBAL VARS
ALPHAVANTAGE_API_KEY = 'T1CY4ZZ01MNPN4YF'


# function
def grabCompact(key, input_ticker, lookBack):

	# Query ALPHA VANTAGE
	try:
	    tS = TimeSeries(key=key, output_format='pandas', indexing_type='date')
	    tI = TechIndicators(key=key, output_format='pandas')

	    data, meta_data = tS.get_daily_adjusted(symbol=input_ticker, outputsize='compact')
	    macd, macd_meta = tI.get_macd(symbol=input_ticker, interval='daily', series_type='close')
	    rsi, rsi_meta = tI.get_rsi(symbol=input_ticker, interval='daily', time_period=14, series_type='close')
	    willr, willr_meta = tI.get_willr(symbol=input_ticker, interval='daily', time_period=14)
	    adx, adx_meta = tI.get_adx(symbol=input_ticker, interval='daily', time_period=14)
	    mom, mom_meta = tI.get_mom(symbol=input_ticker, interval='daily', time_period=10, series_type='close')

	    all_vals = [data, macd, rsi, willr, adx, mom]

	    final_df = pd.concat(all_vals, axis=1, sort=True) # Sort arg may need to be False, leaving it blank raises Pandas error
	    final_df = final_df.dropna()
	    df = final_df.iloc[::-1]
	    df = df.reset_index()
	    df = df.drop(['6. volume', '7. dividend amount'], axis=1)
	    df = df.drop(df.index[lookBack:])

	    return df

	except:
		print ("There was an error with %s" % input_ticker)

		pass

	# ***********************************************


def updateDaily(df, stock):

	# Database Work
	sqlite_dbFile = '/Users/joshuablew/stock.db'

	conn = sqlite3.connect(sqlite_dbFile)
	c = conn.cursor()

	try:
		for i in df.iterrows():
			c.execute("INSERT INTO {tn} VALUES (i[0][1], i[1][1], i[2][1], i[3][1], i[4][1], i[5][1], i[6][1], i[7][1], i[8][1], i[9][1], i[10][1], i[11][1], i[12][1], i[13][1])".\
	        format(tn="AAPL"))

	except Exception as e:
		print (e.message, e.args)

	conn.commit()
	conn.close()


data = grabCompact(ALPHAVANTAGE_API_KEY, "AAPL", 3)
updateDaily(data, "AAPL")



