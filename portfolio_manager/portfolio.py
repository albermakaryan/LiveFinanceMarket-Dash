import pandas as pd
import numpy as np
import yahooquery as yq
import yfinance as yf
import pendulum as pen
import joblib

import time
import datetime as dt
import pendulum as penf

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import tensorflow as tf

from icecream import ic
import json

from portfolio_manager.linear_ts_model_optimization import optimize_arima,optimize_garch
from portfolio_manager.lstm_optimization import create_lstm_model

import os



class Portfolio:


    def __init__(self,watch_list: dict,portfolio_path,records_path,
                 portfolio_performance_path,initial_amount=100_000,
                 best_model_params_path=None,
                 params_from_path=True,model='linear',lstm_dir=None):

        
        self.watch_list = watch_list
        self.portfolio_path = portfolio_path

        self.current_cash = initial_amount
        self.portfolio_value = 0
        self.total_amount = initial_amount

        self.initial_amount = initial_amount

        self.proportions = None
        self.portfolio_performance = None

        self.records_path = records_path
        self.portfolio_performance_path = portfolio_performance_path        
        self.best_model_params_path = best_model_params_path

        self.model = model
        self.lstm_dir = lstm_dir


        self.watch_list_df = pd.DataFrame(self.watch_list.items(),columns=['symbol','stock_name'])
        self.watch_list_df.set_index("symbol",inplace=True)

        if params_from_path:

            try:
                self.best_model_params = pd.read_csv(self.best_model_params_path)
                try:
                        self.best_model_params.set_index("symbol",inplace=True)
                except:
                    pass
            except:
                print("The path is not available!!")
                ask = input("Do you want to optimize models for each stock again? (yes/any other input): ")
                if ask.lower() == 'yes':
                    self.get_best_liner_model(if_exist=False)
                else:
                    print("The optimization interapped!!!")


    def credit(self,amount):

        self.current_cash -= amount

        self.portfolio['cash'] = self.current_cash

        
    def debit(self,amount):

        self.current_cash += amount

        self.portfolio['cash'] = self.current_cash
     
        
    def get_best_liner_model(self,short_run=False,print_summary=False,if_exist=True,save_per_stock=True):
        
        """
        Find the best model for a stock time series to make return prediction for the next day
        
        """

        if if_exist and os.path.exists(self.best_model_params_path):
            ask = input("The best param file exists. Do you want to optimize models for each stock again? (yes/any other input): ")

            if ask.lower() != 'yes':

                self.best_model_params = pd.read_csv(self.best_model_params_path)
                print("The optimization interapted!!")
                return 0
            
            
 
        first_stock = True

        number_of_stocks = len(self.watch_list)
        i = 1
        for ticker in self.watch_list:

            path_to_save_stock_param = "../data/params_per_stock/"+ticker+"_params.csv"

            if save_per_stock and os.path.exists(path_to_save_stock_param):
                continue

            print("\n")
            print("-"*40,self.watch_list[ticker],"  ",round(i/number_of_stocks*100,2),"%",'-'*40)
            print("\n\n")
            i+=1

            
            data = yq.Ticker(ticker).history(period='max')
            try:
                data = data.droplevel(0)
            except:
                pass
            data = data['adjclose'].pct_change(1).dropna().mul(100)

            arima_parameters = optimize_arima(data,test_run=short_run,print_summary=print_summary)
            garch_parameters = optimize_garch(data,test_run=short_run,print_summary=print_summary)


            df_arima = pd.DataFrame(arima_parameters)
            df_arima = df_arima[['parameter']].T.add_prefix("arima_")

            df_garch = pd.DataFrame(garch_parameters)
            df_garch = df_garch[['parameter']].T.add_prefix("garch_")

            df = pd.concat([df_garch,df_arima],axis=1)
            df['symbol'] = ticker
            df['symbol_name'] = self.watch_list[ticker]

            df = df[df.columns[::-1]]

            if first_stock:
                best_model_params = df
                first_stock = False
                continue

            if save_per_stock:
                df.to_csv(path_to_save_stock_param,index=False)

            
            best_model_params = pd.concat([best_model_params,df])


            
            
        best_model_params.set_index("symbol",inplace=True)

        best_model_params.to_csv(self.best_model_params_path)
        self.best_model_params = best_model_params


    def get_best_lstm(self,direcotory_to_save="data/lstm_data",start_date="2015-01-01",train_size=0.8,
                      end_date=None,input_shape=30,batch_size=16,epoch=10,learning_rate=0.03,
                      verbose=1):
        
        
        if os.path.exists(direcotory_to_save):
            ask = input("\nThe directory exists. Do you want to optimize models for each stock again? (yes/any other input): ")

            if ask.lower() != 'yes':
                return 0
        else:
            os.mkdir(direcotory_to_save)
        
        
        return_model_folder,return_scaler_folder = "return_models","return_scalers"
        volatility_model_folder,volatility_scaler_folder = "volatility_models","volatility_scalers"
        
        # create folders if not exists
        
        for folder in [return_model_folder,return_scaler_folder,volatility_model_folder,volatility_scaler_folder]:
            
            path = os.path.join(direcotory_to_save,folder)
            if not os.path.exists(path):
                os.mkdir(path)
        
        for stock in self.watch_list:
            
            
            return_model_path = os.path.join(direcotory_to_save,return_model_folder,stock+".h5")
            return_scaler_path = os.path.join(direcotory_to_save,return_scaler_folder,stock+".pkl")
            volatility_model_path = os.path.join(direcotory_to_save,volatility_model_folder,stock+".h5")
            volatility_scaler_path = os.path.join(direcotory_to_save,volatility_scaler_folder,stock+".pkl")
            
            
            print("\n",self.watch_list[stock],"\nReturn model:")
            
            
            return_model,_,return_scaler = create_lstm_model(stock=stock,input_type='return',
                                                             epoch=epoch,learning_rate=learning_rate,batch_size=batch_size,
                                                             input_shape=input_shape,train_size=train_size,
                                                             start_date=start_date,end_date=end_date,verbose=verbose)
            
            print("\n",self.watch_list[stock],"\nVolatility model:")

            volatility_model,_,volatility_scaler = create_lstm_model(stock=stock,input_type='volatility',
                                                             epoch=epoch,learning_rate=learning_rate,batch_size=batch_size,
                                                             input_shape=input_shape,train_size=train_size,
                                                             start_date=start_date,end_date=end_date,verbose=verbose)
            
            print("\n\n")
            
            return_model.save(return_model_path)
            volatility_model.save(volatility_model_path)
            joblib.dump(return_scaler,filename=return_scaler_path)
            joblib.dump(volatility_scaler,filename=volatility_scaler_path)
            
            

    def get_best_dense(self):
        pass

 
    def sell(self,stock,amount=None):

        price = self.next_day_returns.loc[stock,'last_price']

        # current_portfolio = self.portfolio['portfolio']


        if amount is None:
            stock_sold = self.portfolio['portfolio'][stock]['quantity']
            print(stock_sold)
        
        else:
            stock_sold = amount//price 


        amount_received = stock_sold * price
        stock_name = self.watch_list[stock]

        proportion = amount_received/self.total_amount

        self.debit(amount_received)

        self.portfolio['portfolio_value'] -= amount_received

        if stock not in self.current_best_stocks and stock in self.portfolio['portfolio']:

            # ic('stock')

            self.portfolio['portfolio'].pop(stock)


        else:
            # stock in self.portfolio['portfolio']
            self.portfolio['portfolio'][stock]['quantity'] -= int(stock_sold)
            self.portfolio['portfolio'][stock]['monetary_value'] -= float(amount_received)
            self.portfolio['portfolio'][stock]['proportion'] -= float(proportion)
            self.portfolio['portfolio'][stock]['selling_price'] = float(price)

            # update proportions
            self.proportions[stock] = (self.portfolio['portfolio'][stock]['proportion'],self.portfolio['portfolio'][stock]['monetary_value'])
            try:
                self.portfolio['portfolio'][stock]['average_selling_price'] = float((self.portfolio['portfolio'][stock]['average_selling_price'] + price)/2)
            except:
                self.portfolio['portfolio'][stock]['average_selling_price'] = price

        # if stock in po


        date_time = pen.now()
        date_time = date_time.strftime("%d-%b-%Y %H:%M:%S")

        record  = pd.DataFrame({'date':[date_time],'symbol':[stock],'stock_name':[stock_name],'action':['sell'],'price':[price],'money_amount':[amount_received],
                                'number_of_stocks':[stock_sold]})


        if self.current_records is None:
            self.current_records = record.copy()
        else:
            self.current_records = pd.concat([self.current_records,record])


    def buy(self,stock,amount):   

        


        price,next_return,next_volatility = self.next_day_returns.loc[stock,['last_price','return','volatility']].values.tolist()


        stock_name = self.watch_list[stock]
        stock_bought = amount//price
        amount_spent = price * stock_bought

        proportion = amount_spent/self.total_amount


        self.credit(amount_spent)

        if stock in self.portfolio['portfolio']:
            self.portfolio['portfolio'][stock]['quantity'] += int(stock_bought)
            self.portfolio['portfolio'][stock]['monetary_value'] += float(amount_spent)
            self.portfolio['portfolio'][stock]['proportion'] += float(proportion)
            self.portfolio['portfolio'][stock]['buying_price'] = float(price)
            self.portfolio['portfolio'][stock]['average_buying_price'] = float((self.portfolio['portfolio'][stock]['average_buying_price'] + price)/2)


        else:
            self.portfolio['portfolio'][stock] = {
                "name":stock_name,
                "quantity":int(stock_bought),
                "buying_price":float(price),
                "average_buying_price":float(price),
                "proportion":float(proportion),
                "monetary_value":float(amount_spent),
                "predicted_return":float(next_return),
                "predicted_volatility":float(next_volatility)}
            
        self.portfolio['portfolio_value'] += amount_spent
        # self.portfolio['cash'] -= amount_spent
            
        # update proportions 
        self.proportions[stock] = (self.portfolio['portfolio'][stock]['proportion'],self.portfolio['portfolio'][stock]['monetary_value'])

        # date = pen.today().strftime("%Y-%m-%d")
        date_time = pen.now()
        date_time = date_time.strftime("%d-%b-%Y %H:%M:%S")

        record  = pd.DataFrame({'date':[date_time],'symbol':[stock],'stock_name':[stock_name],'action':['buy'],'price':[price],'money_amount':[amount_spent],
                                'number_of_stocks':[stock_bought]})

        if self.current_records is None:
            self.current_records = record.copy()
        else:
            self.current_records = pd.concat([self.current_records,record])


    def sell_buy_stocks(self):

        total_stock_list = self.current_stocks_in_portfolio.copy()
        total_stock_list.extend(self.current_best_stocks)



        # self.get_current_stocks()

        stock_price = {}

        stocks_definitely_to_sell = [stock for stock in self.current_stocks_in_portfolio if stock not in self.current_best_stocks]



        for stock in total_stock_list:

            if stock in self.current_best_stocks:

                new_proportion,amount = self.proportions[stock]
                amount = int(amount)

            stock_price[stock] = self.next_day_returns.loc[stock,'last_price']

            # new_proportion = proportion

            
            # if some stock are in current portfolio but are not in the best list for current day  sell them
            if stock in stocks_definitely_to_sell:

                ic("Not in the best stock list")

                try:
                    self.current_stocks_in_portfolio.remove(stock)
                except:
                    pass
                
                self.sell(stock)
                continue



            # rechange proportion if stock is bot in current stock list and in portfolio
            elif stock in self.current_stocks_in_portfolio:

                current_proportion = self.portfolio['portfolio'][stock]['proportion']

                ic((new_proportion,current_proportion))

                difference = new_proportion - current_proportion

                amount_to_sell_buy = abs(difference * self.total_amount)


                ic(difference)
                if difference < 0:
                    
                    # ic(amount_to_sell)
                    self.sell(stock,amount_to_sell_buy)

                elif difference > 0:
                    

                    if self.current_cash - amount_to_sell_buy <= 0:
                        print("Not enough money!!!")
                        continue


                    self.buy(stock,amount_to_sell_buy)

                else:
                    continue

                        # if portfolio is emtpy, buy all list
            elif stock not in self.current_stocks_in_portfolio:


                if self.current_cash - amount <= 0:
                    print("Not enough money!!!")
                    continue

                self.buy(stock=stock,amount=amount)

                self.current_stocks_in_portfolio.append(stock)

                


            # if stock is in currebt best stock list but not in the portfolio, buy
            elif stock in self.current_best_stocks  and stock not in self.current_stocks_in_portfolio:
                self.buy(stock,amount)


        # stocks 
        stocks_list = list(stock_price.items())
        stocks_list.sort(reverse=True,key=lambda x: x[1])
        
        ic(self.current_cash)

        for stock,price in stocks_list:
            possible_count_to_buy = self.current_cash//price

            if possible_count_to_buy > 0:
                amount = possible_count_to_buy * price
                self.buy(stock,amount)


        self.portfolio_value = self.portfolio['portfolio_value']
        self.total_amount = self.current_cash + self.portfolio_value
        self.portfolio['total_value'] = self.total_amount

        # update proportsion
        self.proportions['cash'] = self.current_cash/self.total_amount,self.current_cash
             

    def get_return_and_volatility(self,path_to_save=None,model=None,lsmt_dir=None,retrain_lstm=True):

        """
        model: 'linear','lstm','dense-nn'
        """
        
        model = self.model if model is None else model
        lsmt_dir = self.lstm_dir if lsmt_dir is None else lsmt_dir

        date = pen.today().strftime("%Y-%m-%d")


        self.next_day_returns = pd.DataFrame(columns=['return','volatility','last_price','date'],index=self.watch_list_df.index.values)

        for stock in self.watch_list:
            
            ic(stock)
            
            
            
            if model == 'linear':
                
                arima_order = self.best_model_params.loc[stock,['arima_p','arima_d','arima_q']].values

                # convert to sint
                arima_order = list(map(int,arima_order))

                garch_vol,garch_q,garch_p,garch_o,garch_mean,garch_dist = self.best_model_params.loc[stock,['garch_volatility','garch_q','garch_p','garch_o','garch_mean','garch_distribution']].values

                # covert to int
                garch_o,garch_p,garch_q = list(map(int,(garch_o,garch_p,garch_q)))


                data = yq.Ticker(stock).history(period='max')

                last_price = data.tail(1)['adjclose'].values[0]

                data = data.droplevel(0)
                data = data['adjclose'].pct_change(1).dropna().mul(100)

                arima_model = ARIMA(data,order=arima_order)
                arima_result = arima_model.fit()

                garch_model = arch_model(data,vol=garch_vol,mean=garch_mean,dist=garch_dist,q=garch_q,p=garch_p,o=garch_o)
                garch_result = garch_model.fit()

                # predict
                return_forecast = arima_result.forecast().values
                vol_forecast = garch_result.forecast().residual_variance.values[0]

                self.next_day_returns.loc[stock,'return'] = return_forecast
                self.next_day_returns.loc[stock,'volatility'] = vol_forecast
                self.next_day_returns.loc[stock,'last_price'] = last_price
                self.next_day_returns.loc[stock,'date'] = date

        if model == 'lstm':
                
            if not os.path.exists(lsmt_dir):
                ask = input("\nThe director does not exist. Do you want to optimize models for each stock? (yes/any other input): ")
                if ask.lower() != 'yes':
                    return 0
                else:
                    self.get_best_lstm(direcotory_to_save=lsmt_dir)
                    
            elif os.path.exists(lsmt_dir) and retrain_lstm:
                ask = input("\nThe directory exists. Do you want to optimize models for each stock again? (yes/any other input): ")
                if ask.lower() != 'yes':
                    pass
                else:
                    self.get_best_lstm(direcotory_to_save=lsmt_dir)
                              
            for stock in self.watch_list:
                
                print(stock)
                
                for input_type in ['return']:
                    
                    
                    model_path = os.path.join(lsmt_dir,input_type+"_models",stock+".h5")
                    scaler_path = os.path.join(lsmt_dir,input_type+"_scalers",stock+".pkl")
                    
                    model = tf.keras.models.load_model(model_path)
                    
                    with open(scaler_path,'r') as file:
                        scaler = joblib.load(scaler_path)
                        
                        
                    # how many days of data is needed for prediction
                    n_input = model.input.shape[1]
                    
                    # download data from yahoo finance
                    today = dt.datetime.today()
                    n_days_before = today - dt.timedelta(days = 4*n_input)
                    data = yq.Ticker(stock).history(start=n_days_before,end=today)
                    data = data.head(n_input + 1)
                    last_price = data.tail(1)['adjclose'].values[0]
                    
                    if input_type == 'return':

                        data = data['adjclose'].pct_change(1).dropna().values.reshape(-1,1)
                        
                    elif input_type == 'volatility':
                        
                        data = (data['adjclose'].pct_change(1).dropna().rolling(window=252).std() * np.sqrt(252)).dropna().values.reshape(-1,1)

                    data = scaler.transform(data)

                                        
                    prediction = model.predict(data)
                    
                
                    # prediction = prediction[0] #.reshape(-1,1)
                    prediction = scaler.inverse_transform(prediction)
                    prediction = prediction[0][0]
                    self.next_day_returns.loc[stock,input_type] = prediction
                    self.next_day_returns.loc[stock,'volatility'] = 0.05
                    
                    
                    
                self.next_day_returns.loc[stock,'last_price'] = last_price
                self.next_day_returns.loc[stock,'date'] = date
        

            # elif model == 'dense-nn':
            #     pass

            self.next_day_returns.dropna(inplace=True)

            # if path_to_save is not None:
                # self.next_day_returns.to_csv(path_to_save)


    def create_portfolio(self,number_of_stocks=5,update_portfolio=True,model=None,lsmt_dir=None,retrain_lstm=True):

        date = pen.today().strftime("%Y-%m-%d")
        

        date_time = pen.now()
        date_time = date_time.strftime("%d-%b-%Y %H:%M:%S")

        if number_of_stocks > len(self.watch_list):
            raise Exception("number of stocks are more than entire watch list!!!")

        # if exists
        self.get_portfolio_as_dict()


        
        # predicte return and volatility for the next day
        self.get_return_and_volatility(model=model,lsmt_dir=lsmt_dir,retrain_lstm=retrain_lstm)
        
        # ic(self.next_day_returns)
        # quit()
        
        # ic(model)
        # quit()

        # get the bests stocks
        best_stocks = self.next_day_returns.sort_values("return",ascending=False).head(number_of_stocks)
        
        # print(self.next_day_returns)
        # quit()

        # with return more than 0
        # positive_return_stocks = best_stocks[best_stocks['return']>0]
        positive_return_stocks = best_stocks.copy()


        # weight algorithm
    
        positive_return_stocks['return_vol_ratio'] = positive_return_stocks['return']/(positive_return_stocks['return'].sum() + 1e-19) * \
                                                     positive_return_stocks['volatility']/(positive_return_stocks['volatility'].sum() +1e-19 )
        
        positive_return_stocks['weight'] = positive_return_stocks['return_vol_ratio']/(positive_return_stocks['return_vol_ratio'].sum() + 1e-19)

        proportions = positive_return_stocks['weight'].values
        self.current_best_stocks  = positive_return_stocks.index.values.tolist()

        # define recods data frame 
        self.current_records = None
        # self.current_records = pd.DataFrame(['date','symbol','stock_name','action','price','money_amount','number_of_stocks'])

        self.proportions = {stock:[portion,portion*self.total_amount] for stock,portion in zip(self.current_best_stocks, proportions)}

        # peform buy/sell actions
        if update_portfolio:
            self.sell_buy_stocks()
            self.portfolio['date'] = date


        # portfolio performance
        portfolio_performance = pd.DataFrame({"date":[date_time],
                                              "total_amount":[self.total_amount],
                                              "cash":[self.current_cash],
                                              "portfolio_value":[self.portfolio_value],
                                              "portfolio_return":[np.nan]})
        
        if os.path.exists(self.portfolio_performance_path):


            performance_history = pd.read_csv(self.portfolio_performance_path)
            performance_history = pd.concat([performance_history,portfolio_performance])
            performance_history['portfolio_return'] = performance_history['total_amount'].pct_change(1).mul(100)
            performance_history.to_csv(self.portfolio_performance_path,index=False)
            self.portfolio_performance = performance_history

        else:
            portfolio_performance.to_csv(self.portfolio_performance_path,index=False)
            self.portfolio_performance = portfolio_performance




        if self.current_records is not None:

            if os.path.exists(self.records_path):
                full_records = pd.read_csv(self.records_path)
                full_records = pd.concat([full_records,self.current_records])
                full_records.to_csv(self.records_path,index=False)
            else:
                self.current_records.to_csv(self.records_path,index=False)

            self.current_records.set_index("date",inplace=True)





        with open(self.portfolio_path,'w') as f:
            json.dump(self.portfolio,f)


    def get_full_records_as_df(self,return_records=False):

        try:
            self.full_records = pd.read_csv(self.records_path)
        except:
            self.full_records = self.current_records.copy()

        try:
            self.full_records.set_index("date")
        except:
            pass

        if return_records:
            return self.full_records


    def get_portfolio_as_dict(self,return_dict=False):

        if os.path.exists(self.portfolio_path):
            with open(self.portfolio_path,"r") as file:
                self.portfolio = json.load(file)              
                self.current_stocks_in_portfolio = self.portfolio['portfolio']

            self.current_cash = self.portfolio['cash']
            self.portfolio_value = self.portfolio['portfolio_value']
            self.current_stocks_in_portfolio = list(self.portfolio['portfolio'].keys())
            self.total_amount = self.current_cash + self.portfolio_value
        else:
            self.portfolio = {'cash':self.current_cash,"portfolio_value":0,'portfolio':{}}
            self.current_stocks_in_portfolio = []

        if return_dict:
            return self.portfolio


    def get_portfolio_as_df(self,return_df=False):


        df = pd.DataFrame(self.portfolio['portfolio']).T
        df['portfolio_value'] = self.portfolio['portfolio_value']
        df['date'] = self.portfolio['date']
        df['cash'] = self.portfolio['cash']
        df['total_value'] = self.portfolio['total_value']
        df['portfolio_value'] = self.portfolio['portfolio_value']
        df['cash_proportion'] = self.proportions['cash'][0]
        

        self.portfolio_df = df

        if return_df:
            return df

    
    def get_proportions_as_df(self,return_df=False):


        if self.proportions is  None:

            proportions = {}
            portfolio = self.portfolio['portfolio']

            for key in portfolio:
                proportions[key] = portfolio[key]['proportion'],portfolio[key]['monetary_value']
            proportions['cash'] = self.portfolio['cash']/self.portfolio['total_value'],self.portfolio['cash']

            self.proportions = proportions

        df = pd.DataFrame(self.proportions).T
        
        # add name
        df['stock_name'] = np.nan 
        for key in df.index.values:

            if key == 'cash':
                df.loc[key,'stock_name'] = "Cash"
                continue

            df.loc[key,'stock_name'] = self.watch_list[key]

        df.reset_index(inplace=True)
        df.rename(columns={0:"proportion",1:"monetary_value","index":"symbol"},inplace=True)
        self.proportions_df = df

        if return_df:
            return df
        

            # print("creating data frame failed, please check if proportions dict exists or not!!")


    def get_portfolio_performance_as_df(self,return_df=False):


        if os.path.exists(self.portfolio_performance_path):

            if self.portfolio_performance is not None:

                performance_history = pd.read_csv(self.portfolio_performance_path)
                performance_history = pd.concat([performance_history,self.portfolio_performance])
                performance_history['portfolio_return'] = performance_history['total_amount'].pct_change(1).mul(100)
                performance_history.to_csv(self.portfolio_performance_path,index=False)
                self.portfolio_performance = performance_history
            else:
                self.portfolio_performance = pd.read_csv(self.portfolio_performance_path)

        if return_df:
            return self.portfolio_performance


    def get_all_properties(self):
        self.get_portfolio_as_dict()
        self.get_proportions_as_df()
        self.get_portfolio_as_df()
        self.get_full_records_as_df()
        self.get_portfolio_performance_as_df()




