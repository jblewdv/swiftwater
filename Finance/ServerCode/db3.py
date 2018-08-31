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


def getInfo(db1, db2, stock, delta):

    seed = 0
    np.random.seed(seed)
    scaler = MinMaxScaler()

    data1 = pd.read_sql('SELECT * FROM ' + stock, db1)
    data1 = data1.drop(list(range(delta)))

    tableName = stock + 'Labels'
    data2 = pd.read_sql('SELECT * FROM ' + tableName, db2)

    X = data1.iloc[:,1:14]
    labels = data2.iloc[:,1]
    # diffs = data2.iloc[:,2]
    # binLabels = data2.iloc[:,3]

    X[['Open', 'High', 'Low', 'Close', 'AdjClose', 'SplitCoeff', 'MACD', 'MACDhist', 'MACDsignal', 'RSI', 'WILLR', 'ADX', 'MOM']] = scaler.fit_transform(X[['Open', 'High', 'Low', 'Close', 'AdjClose', 'SplitCoeff', 'MACD', 'MACDhist', 'MACDsignal', 'RSI', 'WILLR', 'ADX', 'MOM']])

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, train_size=None, test_size=0.01, shuffle=True, random_state=seed)

    rfc = RandomForestClassifier(n_estimators=100, max_features=5)

    rfc.fit(X_train, Y_train)

    return rfc


def createPickle(rfc, db, stock):

    fileName = stock + "model.pkl"
    path = "pickels/" + stock + "model.pkl"
    tableName = stock + "model"

    joblib.dump(rfc, path)

    cursor = db.cursor()
    
    cursor.execute("CREATE TABLE " + tableName + "(Model VARCHAR(40), FilePath VARCHAR(40))")

    sql = "INSERT INTO " + tableName + "(`Model`, `FilePath`) VALUES (%s, %s)"
    cursor.execute(sql, (stock, path))

    db.commit()



# *************************************************************************

if __name__ == "__main__":
	
    rawinfoDB = pymysql.connect(host='localhost',
                         user='root',
                         password='newpassword',
                         db='rawstockdata',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

    pastpredsDB = pymysql.connect(host='localhost',
                         user='root',
                         password='newpassword',
                         db='stocklabels',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

    pickleDB = pymysql.connect(host='localhost',
                         user='root',
                         password='newpassword',
                         db='pickleStore',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

	# write iteration here...
    cursor = rawinfoDB.cursor()

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='rawstockdata'")

    rows = cursor.fetchall()

    for i in rows:
        stock = i['TABLE_NAME']
        rfc = getInfo(rawinfoDB, pastpredsDB, stock, 7)
        createPickle(rfc, pickleDB, stock)

        print ("Finished %s" % stock)

    print ("Done with job....closing cnxn.")

    rawinfoDB.close()
    pastpredsDB.close()
    pickleDB.close()

# ****************


