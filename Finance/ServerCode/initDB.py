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

# imports
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time
import pymysql


# global variables
# 15 calls/minute
ALPHAVANTAGE_API_KEY = 'NPV7P4MGLYKO6A9S'


# load stock list
data = pd.read_csv('stocks.csv', usecols=[0])
stocks = data['S&P 500 Symbols'].values.tolist()


# get data function
def getData(key, stock):

	tS = TimeSeries(key=key, output_format='pandas', indexing_type='date')
	tI = TechIndicators(key=key, output_format='pandas')

	data, meta_data = tS.get_daily_adjusted(symbol=stock, outputsize='full')
	macd, macd_meta = tI.get_macd(symbol=stock, interval='daily', series_type='close')
	rsi, rsi_meta = tI.get_rsi(symbol=stock, interval='daily', time_period=14, series_type='close')
	willr, willr_meta = tI.get_willr(symbol=stock, interval='daily', time_period=14)
	adx, adx_meta = tI.get_adx(symbol=stock, interval='daily', time_period=14)
	mom, mom_meta = tI.get_mom(symbol=stock, interval='daily', time_period=10, series_type='close')

	all_vals = [data, macd, rsi, willr, adx, mom]

	final_df = pd.concat(all_vals, axis=1, sort=True) # Sort arg may need to be False, leaving it blank raises Pandas error
	final_df = final_df.dropna()
	df = final_df.iloc[::-1]
	df = df.reset_index()
	df = df.drop(['6. volume', '7. dividend amount'], axis=1)

	return df


# add to database function
def add2DB(df, db, stock):

	cursor = db.cursor()

	cursor.execute("CREATE TABLE " + stock + "(id Date, Open float, High float, Low float, Close float, AdjClose float, SplitCoeff int, MACD float, MACDhist float, MACDsignal float, RSI float, WILLR float, ADX float, MOM float)")

	for i in df.iterrows():
		sql = "INSERT INTO " + stock + "(`id`, `Open`, `High`, `Low`, `Close`, `AdjClose`, `SplitCoeff`, `MACD`, `MACDhist`, `MACDsignal`, `RSI`, `WILLR`, `ADX`, `MOM`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
		cursor.execute(sql, (i[1]['index'], i[1]['1. open'], i[1]['2. high'], i[1]['3. low'], i[1]['4. close'], i[1]['5. adjusted close'], i[1]['8. split coefficient'], i[1]['MACD'], i[1]['MACD_Hist'], i[1]['MACD_Signal'], i[1]['RSI'], i[1]['WILLR'], i[1]['ADX'], i[1]['MOM']))
		
		db.commit()

# *************************************************************************

if __name__ == "__main__":
	
	db = pymysql.connect(host='localhost',
			             user='root',
			             password='Isuckatdota2',
			             db='rawstockdata',
			             charset='utf8mb4',
			             cursorclass=pymysql.cursors.DictCursor)

	for index, stock in enumerate(stocks[318:]):
	    df = getData(ALPHAVANTAGE_API_KEY, stock)
	    add2DB(df, db, stock)
	    print ("%s: Finished run.... %s" % (index, stock))
	    print ("...waiting 30 seconds")

	    time.sleep(30)

	print ("Job done...closing database cnxn")
	db.close()

# ****************


