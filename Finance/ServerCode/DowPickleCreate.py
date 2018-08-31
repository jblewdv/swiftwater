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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pymysql
from sklearn.externals import joblib
import time


def run(db1, db2, stock, delta):

    seed = 0
    np.random.seed(seed)
    scaler = MinMaxScaler()

    data1 = pd.read_sql('SELECT * FROM ' + stock, db1)
    data1 = data1.drop(list(range(delta)))

    data2 = pd.read_sql('SELECT * FROM ' + stock, db2)

    X = data1.iloc[:,1:14]
    Y = data2.iloc[:,1]


    X[['Open', 'High', 'Low', 'Close', 'AdjClose', 'SplitCoeff', 'MACD', 'MACDhist', 'MACDsignal', 'RSI', 'WILLR', 'ADX', 'MOM']] = scaler.fit_transform(X[['Open', 'High', 'Low', 'Close', 'AdjClose', 'SplitCoeff', 'MACD', 'MACDhist', 'MACDsignal', 'RSI', 'WILLR', 'ADX', 'MOM']])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=None, test_size=0.1, shuffle=False, random_state=seed)

    rfc = RandomForestClassifier(n_estimators=100, max_features=5)

    rfc.fit(X_train, Y_train)

    rfc.predict(X_test)

    print (rfc.score(X_test, Y_test))

    # fileName = stock + ".pkl"

    # joblib.dump(rfc, fileName)
    # time.sleep(3)

    # rfcLoad = joblib.load(fileName) 

    # data1 = data1.drop('Date', axis=1)

    # preds = rfc.predict(X2[75:])
    # print (rfc.score(X2[75:], labels[75:]))
    # print (preds[0:74])
    



# *************************************************************************

if __name__ == "__main__":
	
    db1 = pymysql.connect(host='localhost',
                         user='root',
                         password='newpassword',
                         db='DowJonesData',
                         charset='utf8',
                         cursorclass=pymysql.cursors.DictCursor)

    db2 = pymysql.connect(host='localhost',
                         user='root',
                         password='newpassword',
                         db='DowJonesLabels',
                         charset='utf8',
                         cursorclass=pymysql.cursors.DictCursor)

    cursor = db1.cursor()

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='DowJonesData' LIMIT 1")

    rows = cursor.fetchall()

    for i in rows:
        stock = i['TABLE_NAME']
        rfc = run(db1, db2, stock, 7)

        print ("Finished %s" % stock)

    print ("Done with job....closing cnxn.")

    db1.close()
    db2.close()

# ****************


