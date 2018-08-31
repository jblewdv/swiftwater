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
import pymysql
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import pprint



def predict(filepath, db1, db2):

    rfc = joblib.load(filepath)

    stock = filepath[8:]
    stock = stock[:-9]

    data = pd.read_sql("SELECT * FROM " + stock + " LIMIT 100", db1)
    dates = data.iloc[:,0]

    data = data.drop('id', axis=1)

    preds = rfc.predict(data).tolist()
    probs = rfc.predict_proba(data).tolist()
    print (preds)
    '''
    cursor = db2.cursor()

    cursor.execute("CREATE TABLE " + stock + "(Date Date, Prediction int, 0_Prob float, 1_Prob float)")
    sql = "INSERT INTO " + stock + "(`Date`, `Prediction`, `0_Prob`, `1_Prob`) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (dates[0], preds[0], probs[0][0], probs[0][1]))

    db2.commit()'''


# *************************************************************************

if __name__ == "__main__":

    rawinfoDB = pymysql.connect(host='localhost',
                         user='root',
                         password='newpassword',
                         db='rawstockdata',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)
	
    pickleDB = pymysql.connect(host='localhost',
                         user='root',
                         password='newpassword',
                         db='pickleStore',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

    predsDB = pymysql.connect(host='localhost',
                         user='root',
                         password='newpassword',
                         db='predictions_7',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

	# write iteration here...
    cursor = pickleDB.cursor()

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='pickleStore'")

    rows = cursor.fetchall()

    for i in rows:
        stock = i['TABLE_NAME']
        if stock == "AAPLmodel":
            cursor.execute("SELECT * FROM " + stock)

            results = cursor.fetchall()

            for i in results:
                path = i['FilePath']
                print ("%s: " % stock)
                predict(path, rawinfoDB, predsDB)
        else:
            pass

    rawinfoDB.close()
    pickleDB.close()
    predsDB.close()

# ****************