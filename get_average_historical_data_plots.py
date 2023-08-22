from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
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
from plot_hourly_data import plot_hourly_data, plot_hourly_mean_and_spread
import yaml
import numpy as np
import ast
from statistics import variance

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

stock_symbol_list = params.get("stock_symbols")
plot_directory = params.get("plot_directory")
data_path = params.get("data_path")
year = params.get("year")

est = pytz.timezone('US/Eastern')
time_axis = [datetime(year, 1, 1, 9,0,0).astimezone(est), datetime(year, 1, 1, 10,0,0).astimezone(est), datetime(year, 1, 1, 11,0,0).astimezone(est),
datetime(year, 1, 1, 12,0,0).astimezone(est), datetime(year, 1, 1, 13,0,0).astimezone(est), datetime(year, 1, 1, 14,0,0).astimezone(est),
datetime(year, 1, 1, 15,0,0).astimezone(est)] #datetime(year, 1, 1, 16,0,0).astimezone(est) #time_axis.append(quote.timestamp.astimezone(est))


for stock_symbol in stock_symbol_list:
    file = open(f"./{data_path}/{stock_symbol}_{year}.csv", "r", newline ='')
    lines = file.readlines()
    file.close()

    time_buckets = [0] * 7 # 9am-4pm
    num_point_in_time_buckets = [0] * 7
    averaged_data = [0] * 7
    timestamps = []
    price_list = []
    value_list = [[],[],[],[],[],[],[]]
    variance_list = []

    # Create list of timestamps and list of prices
    for line in lines:
        data_list = ast.literal_eval(line)
        timestamp = eval(data_list[1])[1]
        ask_price = ast.literal_eval(data_list[3])[1]

        timestamps.append(timestamp)
        price_list.append(ask_price)

    curr_hour = datetime(year, 1, 1, 13, 0, 0)
    curr_hour_bucket = 0

    # Group timestamps and prices into hourly buckets
    for i in range(0, len(timestamps)):
        time = timestamps[i]
        if time.time() >= (curr_hour + timedelta(hours=1)).time():
            curr_hour = (curr_hour + timedelta(hours=1))
            curr_hour_bucket += 1
        if curr_hour_bucket > 6:
            curr_hour = datetime(year, 1, 1, 13, 0, 0)
            curr_hour_bucket = 0

        time_buckets[curr_hour_bucket] += price_list[i]
        num_point_in_time_buckets[curr_hour_bucket] += 1
        value_list[curr_hour_bucket].append(price_list[i])

    # Computes the mean of each hour's data
    averaged_data = np.divide(np.array(time_buckets), np.array(num_point_in_time_buckets)).tolist()

    # Compute the change in price over time


    # Computes the variance of each hour's data
    # for i in range(0, 7):
    #     if len(value_list[i]) <= 1:
    #         variance_list.append(0)
    #     else:
    #         variance_list.append(variance(value_list[i]))
    # price_list[]
    # variance(price_list[0:num_point_in_time_buckets[0]])

    # Plot raw data
    plot_original_data_year(stock_symbol, year, timestamps, price_list, plot_directory)

    # Plot hourly mean data
    plot_hourly_data(stock_symbol, year, time_axis, [averaged_data], plot_directory)

    # Plot hourly mean with spread
    plot_hourly_mean_and_spread(stock_symbol, year, time_axis, value_list, plot_directory)

    # Plot hourly change in value with variance
