from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from datetime import datetime, timedelta, date, time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import pdb
import math
from dates import *
from local_maxima_and_minima import plot_local_maxima_and_minima
from generate_smoothed_data import plot_smoothed_data
from plot_original_data import plot_original_data_day
from plot_hourly_data import plot_hourly_data
import yaml
import numpy as np
import csv
import time



with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

stock_symbol_list = params.get("stock_symbols")
year = params.get("year")
samples_per_day = params.get("samples_per_day")

# tutorialspoint.com/python/time_clock.htm
current_date = datetime(year, 1, 26)
perf_time = 0
process_time = 0
while current_date < datetime(year+1, 1, 1):
    # https://stackoverflow.com/questions/58569361/attributeerror-module-time-has-no-attribute-clock-in-python-3-8
    perf_time = time.perf_counter() - perf_time
    process_time = time.process_time() - process_time

    start_date = current_date + timedelta(hours=13)
    end_date = current_date + timedelta(hours=21)

    request_params = StockQuotesRequest(
                    symbol_or_symbols=stock_symbol_list,
                    start=start_date,
                    end=end_date,
                    ) #limit=len(random_days_sample)*number_of_samples
    try:
        print(f"Date: {current_date}")
        data = client.get_stock_quotes(request_params)
        print(f"Performace Time: {perf_time}")
        print(f"Process Time: {process_time}\n")
    except Exception as e:
        print(f"{e}")

    if data == {}:
        print(f"No data for {current_date}\n")
        current_date = current_date + timedelta(days=1)
        continue


    for stock_symbol in stock_symbol_list:
        data_list = data[stock_symbol]
        spacing = math.ceil(len(data_list)/samples_per_day) # increase length of data/spacing because using more days, 36500
        # List of data points for plotting
        sampled_data_list = data_list[0::spacing]

        file = open(f"{stock_symbol}_{year}", "a", newline ='')

        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows(sampled_data_list)
            # file.close()

    current_date = current_date + timedelta(days=1)
file.close()
