import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time

ALPHAVANTAGE_API_KEY = 'NPV7P4MGLYKO6A9S'


# load stock list
data = pd.read_csv('stocklist.csv', usecols=[0])
stocks = data['STOCKS'].values.tolist()

badOnes = []


def test(key, stock):

	tS = TimeSeries(key=key, output_format='pandas', indexing_type='date')
	data, meta = tS.get_daily_adjusted(symbol=stock, outputsize='compact')


	if (data.shape[0] < 99):
		badOnes.append(stock)
		print ("Bad: %s" % stock)
	else:
		print ("Good: %s" % stock)

	
	



if __name__ == "__main__":
	for i in stocks:
		test(ALPHAVANTAGE_API_KEY, i)
		time.sleep(4)
