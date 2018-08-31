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


# load stock list
data = pd.read_csv('stocks.csv', usecols=[0])
stocks = data['S&P 500 Symbols'].values.tolist()


def queryDB(db, stock, delta):

    data = pd.read_sql('SELECT * FROM ' + stock, db)
   
    labels = []
    priceDiffs = []
    pdLabels = []

    for index, row in data.iterrows():

        if 0 <= index < delta:
            pass

        else:
            initPrice = row[5]
            deltaPrice = data.iloc[index-delta][5]

            if deltaPrice > initPrice:
                labels.append(1)

            else:
                labels.append(0)


            diff = round((deltaPrice-initPrice), 2)
            priceDiffs.append(diff)

            currentDiff = abs(diff)
            threshold = row[5] * 0.01

            if 0 <= currentDiff < threshold:
                pdLabels.append(0)

            else:
                pdLabels.append(1)

    data = data.drop(list(range(delta)))
    data = data[~data.isin(['NaN', 'NaT']).any(axis=1)]
    data = data.reset_index()

    newDF = pd.DataFrame()

    dates = data['id'].values
    newDF['Date'] = pd.Series(dates)
    newDF['Label'] = pd.Series(labels)
    newDF['Price Difference'] = pd.Series(priceDiffs)
    newDF['Bin Label'] = pd.Series(pdLabels)

    return newDF


# add to new db function
def add2DB(df, db, stock):

    tabelName = stock + "Labels"

    cursor = db.cursor()

    cursor.execute("CREATE TABLE " + tabelName + "(id Date, Label int, PriceDiff float, BinLabel int)")

    for i in df.iterrows():

        sql = "INSERT INTO " + tabelName + "(`id`, `Label`, `PriceDiff`, `BinLabel`) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (i[1]['Date'], i[1]['Label'], i[1]['Price Difference'], i[1]['Bin Label']))
        
        db.commit()


# *************************************************************************

if __name__ == "__main__":
	
    dborg = pymysql.connect(host='localhost',
                         user='root',
                         password='Isuckatdota2',
                         db='rawstockdata',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

    dbnew = pymysql.connect(host='localhost',
                         user='root',
                         password='Isuckatdota2',
                         db='stocklabels',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

	# write iteration here...
    # stopped due to error at MKC
    for stock in stocks[318:]:
        result = queryDB(dborg, stock, 7)
        add2DB(result, dbnew, stock)

        print ("Finished run.... %s" % stock)

    print ("Job done...closing database cnxn")
    dborg.close()
    dbnew.close()

# ****************


