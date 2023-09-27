from alpaca.data.requests import StockLatestQuoteRequest, StockQuotesRequest, StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import yaml
import pdb
from datetime import datetime, timezone, date, timedelta
import pytz
import matplotlib.pyplot as plt
import pandas as pd
from determine_trade_times import get_buy_and_sell_signals, get_baseline_signals
from aggregate_data import merge_data, offset_data_by_business_days, get_aggregated_mean
from plot_stock_data import plot

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
data_path = params.get("data_path")
account_data_path = params.get("account_data_path")
dir_name = params.get("rule_based_data_plot_directory")
stock_symbols = params.get("stock_symbols")
year = params.get("year")

start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST
est = pytz.timezone('US/Eastern')

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Get an entire year of data
data_end_date = datetime(year, 12, 31)
data_start_date = datetime(year, 1, 1)

# Uncomment for custom dates
# data_end_date = (datetime.today()+timedelta(days=-10)).replace(hour=16, minute=0, second=0, microsecond=0, tzinfo = est)
# data_start_date = (data_end_date+timedelta(days=-5)).replace(hour=9, minute=0, second=0, microsecond=0, tzinfo = est)

for stock_symbol in stock_symbols:
    data_bars_params = StockBarsRequest(
                    symbol_or_symbols=stock_symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=data_start_date,
                    end=data_end_date.replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
                    )

    # Get data
    data_bars = data_client.get_stock_bars(data_bars_params)
    df = data_bars.df
    df = df.reset_index() # separates symbol and timestamp as columns rather than as multiindex

    # Make sure dataframe only has data for timestamps between 9:30am and 4pm EST
    df = df[(df['timestamp'].dt.tz_convert(est).dt.time >= start_of_trading_day) & (df['timestamp'].dt.tz_convert(est).dt.time <= end_of_trading_day)]
    df.index = df['timestamp']

    # Get Baseline Buy and Sell signals- purely time based
    baseline_buy_signal, baseline_sell_signal = get_baseline_signals(df)

    # Aggregate data for 1-day SMA
    one_day_sma = get_aggregated_mean(df, df.timestamp.dt.date)

    # Offset data by 1 business day to compare each value to previous day's mean
    one_day_sma.timestamp = offset_data_by_business_days(one_day_sma.timestamp, 1)

    # Merge 1-day SMA means with dataframe
    one_day_sma_data = merge_data(df, one_day_sma, 'close', one_day_sma.timestamp.dt.date, df.timestamp.dt.date, 'timestamp')

    # Get Buy and Sell signals for 1-day SMA
    one_day_sma_data = get_buy_and_sell_signals(one_day_sma_data, 'close', 'close_mean')

    # Aggregate data for 1-day prior hourly mean
    hourly_mean = get_aggregated_mean(df, [df.timestamp.dt.date, df.timestamp.dt.hour], index_names=['date', 'hour'])

    # Offset data by 1 business day to compare each value to previous day's mean
    hourly_mean.date = offset_data_by_business_days(hourly_mean.date, 1)

    # Merge hourly mean with dataframe
    hourly_mean_data = merge_data(df, hourly_mean, 'close', [hourly_mean.date.astype('datetime64'), hourly_mean.hour], [df.timestamp.dt.date.astype('datetime64'), df.timestamp.dt.hour], 'timestamp')

    # Get Buy and Sell signals for 1-day hourly mean
    hourly_mean_data = get_buy_and_sell_signals(hourly_mean_data, 'close', 'close_mean')

    # Create plot showing all methods
    plot(x_axis=one_day_sma_data['timestamp'],
        y_axis=[one_day_sma_data['close'], baseline_buy_signal['close'], baseline_sell_signal['close'], one_day_sma_data['buy'],
        one_day_sma_data['sell'], hourly_mean_data['buy'], hourly_mean_data['sell']],
        x_axis_name='Timestamp (EST)',
        y_axis_name='Close Price (USD)',
        title=f"Stock: {stock_symbol} | {year} | Rule Based Methods",
        x_axis_format="date",
        legend_labels=[None, "Baseline SMA Buy", "Baseline SMA Sell", "Daily SMA Buy", "Daily SMA Sell", "Hourly SMA Buy", "Hourly SMA Sell"],
        colors=['#bcbcbc', '#9fc5e8', '#0b5394', '#93c47d', '#38761d', '#ffd966', '#bf9000'],
        marker=[None, '^', 'v', '^', 'v', '^', 'v'],
        linestyle=['-', 'None', 'None', 'None', 'None', 'None', 'None'],
        markersize=2,
        alpha=[None, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        file_name=f"{dir_name}/{stock_symbol}_{year}.svg",
        show_plot=False
        )
