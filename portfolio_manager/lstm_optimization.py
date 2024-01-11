import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import yfinance as yf
import yahooquery as yq
import datetime as dt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import json
import os
from icecream import ic
import warnings
import joblib

warnings.filterwarnings("ignore")


def create_lstm_model(stock='AAPL',start_date="2015-01-01",train_size=0.8,
                      end_date=None,input_shape=30,batch_size=16,epoch=10,learning_rate=0.0001,
                      verbose=1,input_type='return'):
    
    
    """
    input type can be 'return' or 'volatility'
    
    """
    
    end_date = dt.date.today() if end_date is None else end_date
    
    # download data
    data = yq.Ticker(stock).history(start=start_date,end=end_date)
    # data = yq.Ticker(stock).history(period='max')

    
    
    if input_type == 'return':
        data = data['adjclose'].pct_change(1).dropna().values.reshape(-1,1)
        
    elif input_type == 'volatility':
        data = data['adjclose'].pct_change(1).dropna().rolling(window=5).std() * np.sqrt(5)
        data = data.dropna().values.reshape(-1,1)


    print("Number of missing values: ",np.isnan(data).sum())
    
    # split data
    data_size = len(data)
    train_size = int(data_size*train_size)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    
    # data preprocessing
    scaler = StandardScaler()
    scaler = scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    train_data_generator = TimeseriesGenerator(train_data,train_data,length=input_shape,batch_size=batch_size)
    test_data_generator = TimeseriesGenerator(test_data,test_data,length=input_shape,batch_size=batch_size)
    
    

    # build model
    num_features = 1
    
    model = Sequential()
    
    model = Sequential()
    model.add(LSTM(150, activation='tanh', return_sequences=True, input_shape=(input_shape, num_features)))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(64))
    model.add(Dense(1))
    
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    
    # model.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse')
    
    history = model.fit(train_data_generator,epochs=epoch,validation_data=test_data_generator,batch_size=batch_size,verbose=1)
    
    
    return model,history,scaler


if __name__ == "__main__":
    
    model,history,scaler = create_lstm_model(stock='AAPL',start_date="2015-01-01",train_size=0.8,
                      end_date=None,input_shape=30,batch_size=16,epoch=10,learning_rate=0.03,
                      input_type='volatility')
    