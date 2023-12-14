from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta, date, time
import datetime
from datetime import timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import pdb
import math
from dates import *
from local_maxima_and_minima import plot_local_maxima_and_minima
from generate_smoothed_data import plot_smoothed_data
from plot_original_data import plot_original_data_day, plot_original_data_year
from plot_hourly_data import plot_hourly_data, plot_hourly_mean_and_spread # plot_hourly_price_change, plot_hourly_percent_change
import yaml
import numpy as np
import ast
from statistics import variance
from stock_split_info import adjust_for_stock_split

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
stock_symbol_list = params.get("stock_symbols")
plot_directory = params.get("plot_directory")
data_path = params.get("data_path")
year = params.get("year")

est = pytz.timezone('US/Eastern')
time_axis = [datetime(year, 1, 1, 9,0,0).astimezone(est), datetime(year, 1, 1, 10,0,0).astimezone(est), datetime(year, 1, 1, 11,0,0).astimezone(est),
datetime(year, 1, 1, 12,0,0).astimezone(est), datetime(year, 1, 1, 13,0,0).astimezone(est), datetime(year, 1, 1, 14,0,0).astimezone(est),
datetime(year, 1, 1, 15,0,0).astimezone(est), datetime(year, 1, 1, 16,0,0).astimezone(est)] # #time_axis.append(quote.timestamp.astimezone(est))

start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST
start_date = datetime(year, 1, 1)
end_date = datetime(year, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

for stock_symbol in stock_symbol_list:
    data_bars_params = StockBarsRequest(
                    symbol_or_symbols=stock_symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=start_date,
                    end=end_date
                    )

    # Get data
    try:
        data_bars = data_client.get_stock_bars(data_bars_params)
    except Exception as e:
        print(f'Error: {e}')
        print(f'Data unavailable for {stock_symbol}')
        continue

    df = data_bars.df
    df = df.reset_index() # separates symbol and timestamp as columns rather than as multiindex

    # Make sure dataframe only has data for timestamps between 9:30am and 4pm EST
    df = df[(df['timestamp'].dt.tz_convert(est).dt.time >= start_of_trading_day) & (df['timestamp'].dt.tz_convert(est).dt.time <= end_of_trading_day)]
    df.index = df['timestamp']

    hourly_means = df.groupby(df.tz_convert(est).index.hour).mean()

    # df = adjust_for_stock_split(df, stock_symbol, year)

    value_list = [[],[],[],[],[],[],[],[]]
    for count, value in enumerate(df.groupby(df.tz_convert(est).index.hour)):
        value_list[count] = value[1]['close']

    df['pct_change'] = (df['close'].diff() / df['close'].shift(1)) * 100

    # baseline_buy_signal = df.groupby(df.tz_convert(est).index.date, df.tz_convert(est).index.hour).apply(lambda x: x.iloc[[0]]) # first datapoint from each day
    # baseline_sell_signal = data.groupby(data.index.date).apply(lambda x: x.iloc[[-1]]) # last datapoint from each day
    last = df.groupby([df.tz_convert(est).index.date, df.tz_convert(est).index.hour]).last()['close']
    first =  df.groupby([df.tz_convert(est).index.date, df.tz_convert(est).index.hour]).first()['close']

    # Group by hour and calculate the average percentage change
    hourly_average_change = df.groupby(df.tz_convert(est).index.hour)['pct_change'].mean() # looking at the change every minute per hour and averaging that
    # (100*((last-first)/last)).unstack().mean() # Looking at the first and last value from each hour and averaging the change over hour

    hourly_std_dev = df.groupby(df.tz_convert(est).index.hour).std()

    # Plot raw data
    plot_original_data_year(stock_symbol, year, df.index, df['close'], plot_directory, 'svg')

    # Plot hourly mean data with standard deviation
    # plot_hourly_data(stock_symbol, year, time_axis[0:len(hourly_means)], hourly_means['close'].tolist(), plot_directory, 'png', error_bars=hourly_std_dev['close'].tolist(), y_lim=[df['close'].min(), df['close'].max()])
    # plot_hourly_data(stock_symbol, year, time_axis[0:len(hourly_means)], hourly_means['close'], plot_directory, 'png', error_bars=hourly_std_dev['close'], y_lim=[df['close'].min(), df['close'].max()])
    plot_hourly_data(stock_symbol, year, time_axis[0:len(hourly_means)], hourly_means['close'], plot_directory, 'svg', y_lim=[hourly_means['close'].min()-0.5, hourly_means['close'].max()+0.5])
    # hourly_means.index.tolist()

    # Plot hourly mean with spread
    plot_hourly_mean_and_spread(stock_symbol, year, value_list.index, value_list, plot_directory, file_type='svg', y_lim=[df['close'].min(), df['close'].max()])

    # Plot hourly percent change
    plot_hourly_data(stock_symbol, year, time_axis[0:len(hourly_means)], hourly_average_change, plot_directory, 'svg', f'{stock_symbol}_{year}_pct_change', 'Percent Change', y_lim=[hourly_average_change.min()-0.005, hourly_average_change.max()+0.005])
