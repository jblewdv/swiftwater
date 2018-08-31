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

def create_labels(db, stock, delta):

    data = pd.read_sql('SELECT * FROM ' + stock, db)
   
    L = []
    D = []
    bL = []

    for index, row in data.iterrows():

        if (data.shape[0]-delta) <= index < data.shape[0]:
            pass
        else:
            p1 = row[5]
            p2 = data.iloc[index+delta][5]

            if p2 > p1:
                L.append(1)
            else:
                L.append(0)

            diff = round((p2-p1), 2)
            D.append(diff)

            currentDiff = abs(diff)
            threshold = row[5] * 0.01

            if 0 <= currentDiff < threshold:
                bL.append(0)
            else:
                bL.append(1)

    df = pd.DataFrame()

    dates = data['Date'][:-delta].values

    df['Date'] = pd.Series(dates)
    df['Label'] = pd.Series(labels)
    df['Price Difference'] = pd.Series(priceDiffs)
    df['Bin Label'] = pd.Series(pdLabels)

    return df


def add_to_database(df, db, stock):

    cursor = db.cursor()

    cursor.execute("CREATE TABLE " + stock + "(Date Date, Label int, PriceDiff float, BinLabel int, PRIMARY KEY (Date))")

    for i in df.iterrows():

        sql = "INSERT INTO " + stock + "(`Date`, `Label`, `PriceDiff`, `BinLabel`) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (i[1]['Date'], i[1]['Label'], i[1]['Price Difference'], i[1]['Bin Label']))
        
        db.commit()


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

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='DowJonesData'")

    rows = cursor.fetchall()

    for i in rows:
        stock = i['TABLE_NAME']

        df = create_labels(db1, stock, 7)
        add_to_database(df, db2, stock)

        print ("Finished run.... %s" % stock)

    print ("Job done...closing database cnxn")

    db1.close()
    db2.close()

# ****************


