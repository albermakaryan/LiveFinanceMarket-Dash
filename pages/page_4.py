import sys

sys.path.append("../../LiveFinanceMarket-Dash")


# import yfinance as yf
import pandas as pd
import numpy as np

# import plotly.graph_objects as go
# import plotly.io as pio
# import plotly.express as px
from dash import html, Dash, dcc, callback, Input, Output, dash_table
import dash

from icecream import ic
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import json
import datetime as dt

from portfolio_manager.portfolio_creator import PortfolioManager
from portfolio_manager.portfolio_visualizer import PortfolioVisualizer




# load watch list
with open("data/stock/watch_list.json","r") as file:

    watch_list = json.load(file)


# create manager instance of PortfolioManager class to manage portfolio 
manager = PortfolioManager(watch_list=watch_list)
# create visualizer intance of PortfolioVisualizer class to visualize portfolio
visualizer = PortfolioVisualizer()




# linear_portfolio = manager.linear_portfolio()
lstm_portfolio = manager.lstm_portfolio()
manager.update_portfolios(lstm_portfolio)

quit()
# manager.update_portfolio(linear_portfolio)
# manager.get_all_properties(linear_portfolio)

# quit()




CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}




# register page
dash.register_page(__name__,name="Portfolio")

model_tabs = dcc.Tabs(
    children=[
        dcc.Tab(id='linear-portfolio',label='ARIMA & GARCH',value='linear-portfolio'),
        dcc.Tab(id='dense-network-portfolio',label='Dense NN',value='ense-network'),
        dcc.Tab(id='lstm-portfolio',label='LSTM',value='lstm')],
    
    id = 'model-types',
    value='linear-portfolio'
        )


layout = html.Div(

    [

    model_tabs,


    html.Div(
        [

            html.Div([

            ],
            id='pie-chart-containier',
            style={"flex":"50%"}),

              
            
            
            html.Br(),

            html.Div([

            ],
            id = 'info-table')
        ]
    )


    ],
    style=CONTENT_STYLE
)



@callback(
    [Output(component_id='pie-chart-containier',component_property='children'),
    Output(component_id='info-table',component_property='children')],
    Input(component_id='model-types',component_property='value'))


def plot_proportions_and_show_records(portfolio_creation_type):


    now = dt.datetime.now()
    print(now)
    
    model_type = portfolio_creation_type

    # print(model_type)

    if model_type == 'linear-portfolio':
        # print(model_type)
        figure = visualizer.visualize_portfolio(linear_portfolio)
        records = linear_portfolio.get_full_records_as_df(True)

        return html.Div([dcc.Graph(figure=figure)]),\
               html.Div([dash_table.DataTable(data=records.to_dict('records'),columns=[{"name":i,"id":i} for i in records.columns])])
    else:
        # print(model_type)
        return html.Div([]),html.Div([])



# def ff():
#     pass

