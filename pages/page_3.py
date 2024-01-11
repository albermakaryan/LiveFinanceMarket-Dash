import dash
import yfinance as yf
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from dash import html, Dash, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

from portfolio_manager.lstm_optimization import create_lstm_model

valid_intervals = ['1m', '2m', '5m', '15m', '30m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

valid_date_freq = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "90m": "90min",
    "1h": "1h",
    "1d": "1d",
    "5d": "5d",
    "1wk": "1w",
    "1mo": "1m",
    "3mo": "3m"
}


ticker_path = "data/stock/Ticker_list.csv"


symbol_df = pd.read_csv(ticker_path)
forex = symbol_df[symbol_df['Type'] == 'forex']['Symbol'].tolist()

# valid pairs for interval and period
valid_pairs = {
    'max': ['1d', '5d', '1wk', '1mo', '3mo'],
    '2y': ['1h'],
    '1mo': ['90m', '30m', '15m', '5m', '2m'],
    '5d': ['1m']
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
dash.register_page(__name__, name="ARIMA")

layout = html.Div(
    [
        html.H1("Future price prediction with ARIMA",
                style={"textAlign": 'center'}),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Symbol list",
                                style={'textAlign': "center"}),
                        dcc.Dropdown(
                            id='select-symbol',
                            options=symbol_df['Symbol'],
                            value='BTC-USD'
                        ),
                    ],
                    style={'flex': "30%"}

                ),
                html.Div(
                    [
                        html.H3("Timer interval",
                                style={'textAlign': "center"}),
                        dcc.Dropdown(
                            id='time-interval',
                            options=valid_intervals,
                            value='1mo'
                        )
                    ],
                    style={'flex': '30%'}
                ),
                html.Div(
                    [
                        html.H3("N days",
                                style={'textAlign': "center"}),
                        dcc.Slider(
                            id='n-forecast',
                            min=1,
                            max=100,
                            value=20,
                            step=1,
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks=None

                        )
                    ],
                    style={'flex': "40%"}

                )
            ],
            style={'display': 'flex'}

        ),

        html.Div(
            id='model-container'
        ),
    ],

    style=CONTENT_STYLE
)


@callback(
    Output(component_id='model-container', component_property='children'),
    Input(component_id='select-symbol', component_property='value'),
    Input(component_id='time-interval', component_property='value'),
    Input(component_id='n-forecast', component_property='value')
)
def create_model(ticker, interval, n_forecast):
    
    period = [d for d in valid_pairs.keys() if interval in valid_pairs[d]][0]

    if ticker in forex:
        df = yf.Ticker(ticker + '=X').history(period=period, interval=interval).reset_index()
    else:
        df = yf.Ticker(ticker).history(period=period, interval=interval).reset_index()

    df['Date'] = pd.to_datetime(df.iloc[:, 0])

    # calculating moving average
    df['MA'] = df['Close'].rolling(7).mean()
    df['Mean'] = (df['Close'] + df['Open']) / 2

    model_df = df[['Date', 'Mean']].copy()
    model_df['Type'] = 'Historical'
    # model_df['Type'][-1] = 'Prediction'
    # model_df['Type'][-2] = 'Prediction'

    model = ARIMA(df['Mean'], order=(1, 0, 1))
    result = model.fit()
    prediction_result = result.get_forecast(steps=n_forecast, alpha=0.99)

    prediction = prediction_result.predicted_mean

    date_range = pd.date_range(start=max(df['Date']),
                               periods=n_forecast,
                               freq=valid_date_freq[interval])

    prediction_df = pd.DataFrame(
        {"Date": date_range,
         "Mean": prediction,
         "Type": "Prediction"}
    )

    conf_interval = prediction_result.conf_int()

    conf_interval['Date'] = date_range
    plot_df = pd.concat([model_df, prediction_df])

    figure = px.line(
        data_frame=plot_df,
        x='Date',
        y='Mean',
        color='Type'
    )

    figure.add_trace(
        go.Scatter(
            x=conf_interval['Date'],
            y=conf_interval['upper Mean'],
            name='Upper'
        )
    )
    figure.add_trace(
        go.Scatter(
            x=conf_interval['Date'],
            y=conf_interval['lower Mean'],
            name='Lower'
        )
    )

    figure.update_layout(
        title=ticker + str(n_forecast) + " days prediction",
        template=pio.templates['plotly_white']

    )

    return html.Div(
        [
            dcc.Graph(
                figure=figure
            )
        ]
    )
