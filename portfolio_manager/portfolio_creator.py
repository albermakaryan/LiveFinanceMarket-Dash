import sys 
# add project path to be able to import portfolio_manager module

sys.path.append("../../LiveFinanceMarket-Dash")



from icecream import ic

import json

from portfolio_manager.portfolio import Portfolio

# ic("Done")




# def linear_portfolio():

import pandas as pd

class PortfolioManager:

    def __init__(self,watch_list: dict):

        self.watch_list = watch_list

        self.portfolio_list = {}
        pass


    def linear_portfolio(self,
                        n_scotks = 5,
                        BEST_LINEAR_MODEL_PARAM_PATH = 'data/model_data/best_linear_model_params.csv',
                        BEST_LINEAR_MODEL_PARAM_PATH_FULL = 'data/model_data/best_linear_model_params_full.csv',
                        LINEAR_PORTOFLIO_PATH = 'data/portfolio_data/linear_portfolio/linear_portfolio.json',
                        LINEAR_RETURNS_PATH = 'data/portfolio_data/linear_portfolio/linear_returns.csv',
                        LINEAR_RECORDS_PATH = 'data/portfolio_data/linear_portfolio/linear_records.csv',
                        PORTFOLIO_PERFORMACE_PATH = 'data/portfolio_data/linear_portfolio/linear_portfolio_perfomance.json',
                        model='linear'):
        
        portfolio = Portfolio(watch_list=self.watch_list,
                                   initial_amount=100_000,
                                   best_model_params_path=BEST_LINEAR_MODEL_PARAM_PATH,
                                   portfolio_path=LINEAR_PORTOFLIO_PATH,
                                   records_path=LINEAR_RECORDS_PATH,params_from_path=True,
                                   portfolio_performance_path=PORTFOLIO_PERFORMACE_PATH,
                                   model=model)
        
        self.portfolio_list['linear-portfolio'] = portfolio
        

        return portfolio
    
    def lstm_portfolio(self,
                    n_scotks = 5,
                    LSTM_PORTOFLIO_PATH = 'data/portfolio_data/lstm_portfolio/lstm_portfolio.json',
                    LSTM_RETURNS_PATH = 'data/portfolio_data/lstm_portfolio/lstm_returns.csv',
                    LSTM_RECORDS_PATH = 'data/portfolio_data/lstm_portfolio/lstm_records.csv',
                    PORTFOLIO_PERFORMACE_PATH = 'data/portfolio_data/lstm_portfolio/lstm_portfolio_perfomance.json',
                    model='lstm',lstm_dir="data/lstm_data"):
    
        portfolio = Portfolio(watch_list=self.watch_list,
                                initial_amount=100_000,
                                portfolio_path=LSTM_PORTOFLIO_PATH,
                                records_path=LSTM_RECORDS_PATH,params_from_path=False,
                                portfolio_performance_path=PORTFOLIO_PERFORMACE_PATH,
                                model=model,lstm_dir=lstm_dir)
        
        self.portfolio_list['lstm-portfolio'] = portfolio
        

        return portfolio
    
    def get_all_properties(self,portfolio=None):
        
        if portfolio is None:
            for portfolio in self.portfolio_list:
                
                self.portfolio_list[portfolio].get_all_properties()
        else:

            portfolio.get_all_properties()


    def update_portfolios(self,portfolio=None,n_stocks=5):
        
        if portfolio is None:
            for portfolio in self.portfolio_list:
                model = portfolio.split('-')[0]
                self.portfolio_list[portfolio].create_portfolio(model=model,number_of_stocks=n_stocks,update_portfolio=True)
        else:
            model = portfolio.split('-')[0]
            self.portfolio_list[portfolio].create_portfolio(model=model,number_of_stocks=n_stocks,update_portfolio=True)





# BEST_LINEAR_MODEL_PARAM_PATH = '../data/model_data/best_linear_model_params.csv'
# LINEAR_PORTOFLIO_PATH = '../data/portfolio_data/linear_portfolio.json'
# LINEAR_RETURNS_PATH = '../data/portfolio_data/linear_returns.csv'

# with open("../data/stock/watch_list.json",'r') as file:

#     watch_list = json.load(file)





# linear_portfolio = Portfolio(tickers=watch_list,initial_amount=100_000,best_model_params_path=BEST_LINEAR_MODEL_PARAM_PATH,params_from_path=True)

# linear_portfolio.create_portfolio(current_portfolio_json_path=LINEAR_PORTOFLIO_PATH,current_returns_path=LINEAR_RETURNS_PATH)

# print(linear_portfolio.portfolio)

# params = pd.read_csv("../data/model_data/best_model_params.csv")

# # print(params.columns)
 
  



# # print(portfolio.best_model_params)

# # portfolio.get_best_liner_model(short_run=True)

# # print(portfolio.next_day_returns)

# # print(type(pm.Portfolio))
# ic("Done!!!")

