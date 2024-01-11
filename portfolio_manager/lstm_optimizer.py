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




def lstm_model_opmizer(stock,end=None,train_size = 0.8,epoch=1):

    
    end = dt.datetime.today() if end is None else end
    
    
    # download data from yahoo finance
    data = yq.Ticker(stock).history(start="2021-01-01",end=end)

    data = data['adjclose'].pct_change(1).dropna().values.reshape(-1,1)
    
    data_size = len(data)
    
    # prepare data for lstm model
    train_size = int(data_size*train_size) #if train_size <=1 else int(train_size)
    
    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = StandardScaler()
    scaler = scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    
    # hypermaters
    n_input = 30
    batch_size = 16
    LR = 0.03
    num_features = 1
    epochs = epoch
    
    # data preprocessing
    train_data_generator = TimeseriesGenerator(train_data,train_data,length=n_input,batch_size=batch_size)
    test_data_generator = TimeseriesGenerator(test_data,test_data,length=n_input,batch_size=batch_size)
    
    
    # build lstm model


    model = Sequential()

    model.add(LSTM(128,input_shape=(n_input,num_features),return_sequences=True,activation='relu'))
    # model.add(Dropout(0.2))
    model.add(LSTM(128,input_shape=(n_input,num_features),return_sequences=True,activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(LSTM(128,input_shape=(n_input,num_features),return_sequences=True,activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    # model.add(Dense(64,activation='relu'))
    model.add(Dense(1))
    
    
    for batch in train_data_generator:
        x,y = batch
        ic(x.shape)
        ic(y.shape)
        break
    
    
    model.compile(optimizer=Adam(learning_rate=LR),loss='mse')
    

    
    
    history = model.fit(train_data_generator,epochs=epochs,validation_data=test_data_generator,batch_size=batch_size,verbose=1)
    

    # output
    output = {
        "model":model,
        "scaler":scaler,
        "n_input":n_input,
        "history":history,
    }
    
    return model,history,scaler



def lstm_for_all_stocks(watch_list,directory,model_folder,scaler_folder,epoch=1):
    
    
    [os.makedirs(os.path.join(directory,path)) for path in [model_folder,scaler_folder] if not os.path.exists(os.path.join(directory,path))]
    


    
    for stock in watch_list:
        
        model,_,scaler = lstm_model_opmizer(stock,epoch=epoch)
        
        model_path = os.path.join(directory,model_folder,stock+".h5")
        scaler_path = os.path.join(directory,scaler_folder,stock+".pkl")
        
                
        model.save(model_path)
        joblib.dump(scaler,filename=scaler_path)
    
    
def load_lstm_model(stock,directory,model_folder,scaler_folder):
    
    
    
    models = os.listdir(os.path.join(directory,model_folder))
    scalers = os.listdir(os.path.join(directory,scaler_folder))
    

    
    models = {model.split(".")[0]:os.path.join(directory,model_folder,model) for model in models}
    scalers = {scaler.split(".")[0]:os.path.join(directory,scaler_folder,scaler) for scaler in scalers}
    
    
    # for stock in models:
    #     ic(models[stock])
    #     print()
    # quit()



    # print(models)
    # print(scalers)
    
    # quit()
    
    for stock in models:
        
        
        model_path = models[stock]
        scaler_path = scalers[stock]
        
        # ic(model_path)
        # continue
        model = tf.keras.models.load_model(model_path)
        
        

        
        with open(scaler_path,'r') as file:
            scaler = joblib.load(scaler_path)
        
        
        # how many days of data is needed for prediction
        n_input = model.input.shape[1]
        
        # current date
        today = dt.datetime.today()
        n_days_before = today - dt.timedelta(days = 4*n_input)
        
        # dowlnoad data from yahoo finance
        data = yq.Ticker(stock).history(start=n_days_before,end=today)
        data = data.head(31)
        data = data['adjclose'].pct_change(1).dropna().values.reshape(-1,1)
        data = scaler.transform(data)
        
        
        prediction = model.predict(data).reshape(-1,1)
        prediction = scaler.inverse_transform(prediction)
        prediction = prediction[0,0]
        
        print(f"stock: {stock}, next day prediction prediction: {prediction}")
        print()
        # scale data
        
        
        
      
        
    
    
    # print("\n\n")
    # print(models)
    # print(models)

    
    
    
    
    
def lstm_prediction():
    
    pass    
    
    # print(train_data.shape)
    # print(test_data.shape)

if __name__ == "__main__":
    
    
    
    
    with open("../data/stock/watch_list.json","r") as file:
        watch_list = json.load(file)
        
        
        
        
    directory = "../data/lstm_data"
    model_folder = "models"
    scaler_folder = "scalers"
        
        
    stock = "AAPL"
    
    
    # lstm_model_opmizer(stock)
    
    lstm_for_all_stocks(watch_list,directory=directory,model_folder=model_folder,scaler_folder=scaler_folder)
    
    #
    load_lstm_model(stock,directory,model_folder=model_folder,scaler_folder=scaler_folder)