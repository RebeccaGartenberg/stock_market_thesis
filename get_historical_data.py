from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import pdb
import math
from dates import *
from local_maxima_and_minima import plot_local_maxima_and_minima
from generate_smoothed_data import plot_smoothed_data
from plot_original_data import plot_original_data_day
import yaml

with open('/Users/rebeccagartenberg/stock_market_analysis/input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")

stock_symbol_list = params.get("stock_symbols")
all_dates = params.get("all_dates")
n_dates = params.get("total_dates")
number_of_samples = params.get("total_samples")
window_sizes = params.get("window_sizes", [])

plot_directory = params.get("plot_directory")

random_days_sample = get_n_random_dates(datetime(2022, 1, 1), datetime(2022, 12, 31), n_dates, all_dates)

time_axis = [0] * number_of_samples
price_axis = [0] * number_of_samples
for stock_symbol in stock_symbol_list:
    for rand_date in random_days_sample:
        start_date = rand_date + timedelta(hours=13)
        end_date = rand_date + timedelta(days=1)

        client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        request_params = StockQuotesRequest(
                        symbol_or_symbols=stock_symbol,
                        start=start_date,
                        end=end_date)
        data = client.get_stock_quotes(request_params)
        stock_price_list = []

        data_list = data[stock_symbol]
        spacing = math.ceil(len(data_list)/number_of_samples)
        sampled_data_list = data_list[0::spacing]  #try 250 points

        est = pytz.timezone('US/Eastern')

        # For plotting the data by day
        time_axis = []
        price_axis = []
        for quote in sampled_data_list:
            time_axis.append(quote.timestamp.astimezone(est))
            price_axis.append(quote.ask_price)

        plot_original_data_day(stock_symbol, start_date, time_axis, price_axis, plot_directory)

        plot_local_maxima_and_minima(stock_symbol, sampled_data_list, start_date, time_axis, price_axis, plot_directory)

        for window in window_sizes:
            plot_smoothed_data(stock_symbol, start_date, time_axis, price_axis, window, plot_directory)

    plot_original_data_day(stock_symbol, start_date, time_axis, price_axis/len(random_days_sample), plot_directory)
