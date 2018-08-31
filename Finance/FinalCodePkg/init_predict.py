
# imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pymysql


seed = 52
np.random.seed(seed)
scaler = MinMaxScaler()


def predict(db, db2, stock, delta):

    frames_back = 1

    prices = pd.read_sql('SELECT * FROM ' + stock, db)
    prices = prices.drop(list(range(prices.shape[0]-(delta-frames_back), prices.shape[0])))
    labels = pd.read_sql('SELECT * FROM ' + stock, db2)

    indexList = list(prices.index.values)
    current = indexList[-frames_back]

    X = prices.iloc[:,1:14]
    labels = labels.iloc[:,1]
    labels = labels.to_frame()

    d = {'Label': [0]}
    df = pd.DataFrame(data=d)

    labels = pd.concat([labels, df], ignore_index=True)

    X[['Open', 'High', 'Low', 'Close', 'AdjClose', 'SplitCoeff', 'MACD', 'MACDhist', 'MACDsignal', 'RSI', 'WILLR', 'ADX', 'MOM']] = scaler.fit_transform(X[['Open', 'High', 'Low', 'Close', 'AdjClose', 'SplitCoeff', 'MACD', 'MACDhist', 'MACDsignal', 'RSI', 'WILLR', 'ADX', 'MOM']])

    x_train, x_test, y_train, y_test = train_test_split(X, labels, train_size=None, test_size=0.5, shuffle=True, random_state=seed)


    trainIndex = list(x_train.index.values)
    testIndex = list(x_test.index.values)

    if current in testIndex:
        print ("Found in test set")

        rfc = RandomForestClassifier(n_estimators=100, max_features=5)
        rfc.fit(x_train, y_train.values.ravel())
        preds = rfc.predict(x_test)
        probs = rfc.predict_proba(x_test).tolist()
        score = rfc.score(x_test, y_test)

        counter = 0
        for index, row in x_test.iterrows():
            if index == current:
                break
            else:
                counter +=1

        print(score)
        # return preds[counter], probs[counter]
        print (preds[counter], probs[counter])

    else:
        print ("Found in train set")

        rfc = RandomForestClassifier(n_estimators=100, max_features=5)
        rfc.fit(x_test, y_test.values.ravel())
        preds = rfc.predict(x_train)
        probs = rfc.predict_proba(x_train).tolist()
        score = rfc.score(x_train, y_train)

        counter = 0
        for index, row in x_train.iterrows():
            if index == current:
                break
            else:
                counter +=1

        print(score)
        # return preds[counter], probs[counter]
        print (preds[counter], probs[counter])




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

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='DowJonesData' LIMIT 8")

    rows = cursor.fetchall()

    for i in rows:
        stock = i['TABLE_NAME']
        predict(db1, db2, stock, 7)
       

        print ("Finished %s" % stock)

    print ("Done with job....closing cnxn.")

    db1.close()
    db2.close()
    

# ****************