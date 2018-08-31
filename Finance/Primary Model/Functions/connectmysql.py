# IMPORTS
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from time import sleep
import pymysql.cursors



def add2Database(df, stock):
	$stock = stock

	try:
		db = pymysql.connect(host='localhost',
		                     user='root',
		                     password='Isuckatdota2',
		                     db='RawStockDataTables',
		                     charset='utf8mb4',
		                     cursorclass=pymysql.cursors.DictCursor)

	    c = db.cursor()
	    
	    sqlQuery = "CREATE TABLE $stock (id int, LastName varchar(32), FirstName varchar(32), DepartmentCode int)"   

	    c.execute(sqlQuery)

	    #c.commit()

	finally:
	    db.close()



# *******************

'''data = pd.read_csv('stocks.csv', usecols=[0])
stocks = data['S&P 500 Symbols'].values.tolist()


for stock in stocks:
    df = getRawData(ALPHAVANTAGE_API_KEY, stock)
    add2Database(df, stock)
    print ("Finished run.... %s" % stock)
    print ("...wait 15 seconds")

    sleep(15)'''

# *******************