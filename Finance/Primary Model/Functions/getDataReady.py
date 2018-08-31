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
# *************************************************************************

# IMPORTS
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

# *************************************************************************

seed = 0
np.random.seed(seed)

ALPHAVANTAGE_API_KEY = 'T1CY4ZZ01MNPN4YF'

def grabData(key, input_ticker, delta):
    tS = TimeSeries(key=key, output_format='pandas', indexing_type='date')
    tI = TechIndicators(key=key, output_format='pandas')

    data, meta_data = tS.get_daily_adjusted(symbol=input_ticker, outputsize='full') #compact for last 100 or full for everything
    macd, macd_meta = tI.get_macd(symbol=input_ticker, interval='daily', series_type='close')
    rsi, rsi_meta = tI.get_rsi(symbol=input_ticker, interval='daily', time_period=14, series_type='close')
    willr, willr_meta = tI.get_willr(symbol=input_ticker, interval='daily', time_period=14)
    adx, adx_meta = tI.get_adx(symbol=input_ticker, interval='daily', time_period=14)
    mom, mom_meta = tI.get_mom(symbol=input_ticker, interval='daily', time_period=10, series_type='close')

    all_vals = [data, macd, rsi, willr, adx, mom]

    final_df = pd.concat(all_vals, axis=1, sort=True) # Sort arg may need to be False, leaving it blank raises Pandas error
    final_df = final_df.dropna()
    df = final_df.iloc[::-1]
    df = df.reset_index()
    df = df.drop(['6. volume', '7. dividend amount'], axis=1)

    # *************************************************************************
    
    labels = []
    priceDiffernces = []

    for index, row in df.iterrows():
        if 0 <= index < delta:
            pass
        else:
            initPrice = row[5]
            deltaPrice = df.iloc[index-delta][5]
            priceDiffernces.append(round((deltaPrice-initPrice), 2))

            if deltaPrice > initPrice:
                labels.append(1)
            else:
                labels.append(0)
    
    df = df.drop(list(range(delta)))

    labelsPD = pd.Series(labels)
    priceDiffsPD = pd.Series(priceDiffernces)
    df['Labels'] = labelsPD.values
    df['Price Diffs'] = priceDiffsPD.values

    df.to_csv(input_ticker + '.csv')
 
    return df
# *************************************************************************

if __name__ == "__main__":
    grabData(ALPHAVANTAGE_API_KEY, "AAPL", 7)
    #print (output)

# *************************************************************************
