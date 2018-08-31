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



#ALPHAVANTAGE_API_KEY = 'T1CY4ZZ01MNPN4YF'
ALPHAVANTAGE_API_KEY = 'REJVUC8NB244EQ9T'

def getRawData(key, input_ticker):

	try:
	    tS = TimeSeries(key=key, output_format='pandas', indexing_type='date')
	    tI = TechIndicators(key=key, output_format='pandas')

	    data, meta_data = tS.get_daily_adjusted(symbol=input_ticker, outputsize='full') #compact for last 100 or full for everything
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

	    return df

	except:
		print ("There was an error with %s" % input_ticker)
		pass

    
def add2Database(df, stockName):

	con = sqlite3.connect("/Users/joshuablew/stock.db")
	df.to_sql(stockName, con, if_exists='replace', index=False)

	# ???
	#con.commit()
	#con.close()


# *******************

'''data = pd.read_csv('stocks.csv', usecols=[0])
stocks = data['S&P 500 Symbols'].values.tolist()


for stock in stocks:
    df = getRawData(ALPHAVANTAGE_API_KEY, stock)
    add2Database(df, stock)
    print ("Finished run.... %s" % stock)
    print ("...wait 15 seconds")

    sleep(15)
'''
out = getRawData(ALPHAVANTAGE_API_KEY, "AAPL")
print (out)
# *******************
