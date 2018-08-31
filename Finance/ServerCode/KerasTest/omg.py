import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Function
def go(db1, db2, stock, delta):

    data = pd.read_sql('SELECT * FROM ' + stock, db1)
    data = data.drop(list(range(delta)))

    labels = pd.read_sql('SELECT * FROM ' + stock, db2)

    X = data1.iloc[:,1:14]
    Y = data2.iloc[:,1]

    print(X)
    print (Y)


    # model = Sequential()
    # model.add(Dense(64, input_dim=20, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    # model.fit(x_train, y_train,
    #           epochs=20,
    #           batch_size=128)
    # score = model.evaluate(x_test, y_test, batch_size=128)



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
        rfc = getInfo(db1, db2, stock, 7)

        print ("Finished %s" % stock)

    print ("Done with job....closing cnxn.")

    db1.close()
    db2.close()

# ****************


