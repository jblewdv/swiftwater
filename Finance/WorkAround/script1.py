
# imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pymysql


def getInfo(db1, db2, stock, delta):

    seed = 52
    np.random.seed(seed)
    scaler = MinMaxScaler()

    data1 = pd.read_sql('SELECT * FROM ' + stock, db1)
    data1 = data1.drop(list(range(delta)))

    data2 = pd.read_sql('SELECT * FROM ' + stock, db2)

    X = data1.iloc[:,1:14]
    labels = data2.iloc[:,1]

    X[['Open', 'High', 'Low', 'Close', 'AdjClose', 'SplitCoeff', 'MACD', 'MACDhist', 'MACDsignal', 'RSI', 'WILLR', 'ADX', 'MOM']] = scaler.fit_transform(X[['Open', 'High', 'Low', 'Close', 'AdjClose', 'SplitCoeff', 'MACD', 'MACDhist', 'MACDsignal', 'RSI', 'WILLR', 'ADX', 'MOM']])

    x_train, x_test, y_train, y_test = train_test_split(X, labels, train_size=None, test_size=0.5, shuffle=True, random_state=seed)

    y_train = y_train.to_frame()
    y_test = y_test.to_frame()
    
    testIndex = list(x_test.index.values)
    indexSearch = 7

    if indexSearch in testIndex:
        print ("Sample is in test set...initializing model")

        rfc = RandomForestClassifier(n_estimators=100, max_features=5)
        rfc.fit(x_train, y_train.values.ravel())
        preds = rfc.predict(x_test)
        probs = rfc.predict_proba(x_test).tolist()
        score = rfc.score(x_test, y_test)

        counter = 0
        for index, row in x_test.iterrows():
            if index == indexSearch:
                break
            else:
                counter +=1

        return preds[counter], probs[counter]

    else:
        print ("Sample seems to be in train set...initializing inverse model")

        rfc = RandomForestClassifier(n_estimators=100, max_features=5)
        rfc.fit(x_test, y_test.values.ravel())
        preds = rfc.predict(x_train)
        probs = rfc.predict_proba(x_train).tolist()
        score = rfc.score(x_train, y_train)

        counter = 0
        for index, row in x_test.iterrows():
            if index == indexSearch:
                break
            else:
                counter +=1

        return preds[counter], probs[counter]
        # finish edit

       


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

    # db3 = pymysql.connect(host='localhost',
    #                      user='root',
    #                      password='newpassword',
    #                      db='DowJonesPredictions',
    #                      charset='utf8',
    #                      cursorclass=pymysql.cursors.DictCursor)

	# write iteration here...
    cursor = db1.cursor()

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='DowJonesData' LIMIT 1")

    rows = cursor.fetchall()

    for i in rows:
        stock = i['TABLE_NAME']
        pred, prob = getInfo(db1, db2, stock, 7)
        print (pred, prob)

        print ("Finished %s" % stock)

    print ("Done with job....closing cnxn.")

    db1.close()
    db2.close()
    

# ****************


