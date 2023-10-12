from alpaca.data.requests import StockLatestQuoteRequest, StockQuotesRequest, StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import yaml
import pdb
from datetime import datetime, timezone, date, timedelta
import pytz
import matplotlib.pyplot as plt
import pandas as pd
from determine_trade_times import get_buy_and_sell_signals, get_baseline_signals, \
    get_buy_and_sell_signals_ROC, get_buy_and_sell_signals_combined
from aggregate_data import merge_data, offset_data_by_business_days, get_aggregated_mean, get_aggregated_mean_hourly
from plot_stock_data import plot
from analyze_trades import determine_profits, get_total_trades, get_total_trades_per_hour, get_total_profits_per_hour
import dataframe_image as dfi
from stock_split_info import adjust_for_stock_split

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
data_path = params.get("data_path")
account_data_path = params.get("account_data_path")
dir_name = params.get("rule_based_data_plot_directory")
tables_dir_name = params.get("rule_based_tables_directory")
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

best_strategies = {}
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

    df = adjust_for_stock_split(df, stock_symbol, year)

    # Get Baseline Buy and Sell signals- purely time based
    baseline_buy_signal, baseline_sell_signal = get_baseline_signals(df)

    # Aggregate data
    one_day_sma = get_aggregated_mean(df, df.timestamp.dt.date) # 1-day SMA
    two_day_sma = get_aggregated_mean(df, df.timestamp.dt.date, 2) # 2-day SMA
    three_day_sma = get_aggregated_mean(df, df.timestamp.dt.date, 3) # 3-day SMA
    ten_day_sma = get_aggregated_mean(df, df.timestamp.dt.date, 10) # 10-day SMA
    twenty_day_sma = get_aggregated_mean(df, df.timestamp.dt.date, 20) # 10-day SMA

    # Offset data by 1 business day to compare each value to previous day's mean
    one_day_sma.timestamp = offset_data_by_business_days(one_day_sma.timestamp, 1)
    two_day_sma.timestamp = offset_data_by_business_days(two_day_sma.timestamp, 1)
    three_day_sma.timestamp = offset_data_by_business_days(three_day_sma.timestamp, 1)
    ten_day_sma.timestamp = offset_data_by_business_days(ten_day_sma.timestamp, 1)
    twenty_day_sma.timestamp = offset_data_by_business_days(twenty_day_sma.timestamp, 1)

    # Merge n-day SMA means with dataframe
    one_day_sma_data = merge_data(df, one_day_sma, 'close', one_day_sma.timestamp.dt.date, df.timestamp.dt.date, True, 'timestamp')
    two_day_sma_data = merge_data(df, two_day_sma, 'close', two_day_sma.timestamp.dt.date, df.timestamp.dt.date, True, 'timestamp')
    three_day_sma_data = merge_data(df, three_day_sma, 'close', three_day_sma.timestamp.dt.date, df.timestamp.dt.date, True, 'timestamp')
    ten_day_sma_data = merge_data(df, ten_day_sma, 'close', ten_day_sma.timestamp.dt.date, df.timestamp.dt.date, True, 'timestamp')
    twenty_day_sma_data = merge_data(df, twenty_day_sma, 'close', twenty_day_sma.timestamp.dt.date, df.timestamp.dt.date, True, 'timestamp')

    # Get Buy and Sell signals for 1-day SMA
    one_day_sma_data = get_buy_and_sell_signals(one_day_sma_data, 'close', 'close_mean')
    two_day_sma_data = get_buy_and_sell_signals(two_day_sma_data, 'close', 'close_mean')
    three_day_sma_data = get_buy_and_sell_signals(three_day_sma_data, 'close', 'close_mean')
    ten_day_sma_data = get_buy_and_sell_signals(ten_day_sma_data, 'close', 'close_mean')
    twenty_day_sma_data['ten_day_sma'] = ten_day_sma_data['close_mean']
    crossover = get_buy_and_sell_signals(twenty_day_sma_data, 'close_mean', 'ten_day_sma')

    hourly_mean = get_aggregated_mean_hourly(df, 2)
    hourly_mean_10_day = get_aggregated_mean_hourly(df, 10)
    hourly_mean_20_day = get_aggregated_mean_hourly(df, 20)

    # Add offset by 1 day here
    hourly_mean['timestamp'] = hourly_mean.index
    hourly_mean.timestamp = offset_data_by_business_days(hourly_mean.timestamp, 1)
    hourly_mean_10_day['timestamp'] = hourly_mean_10_day.index
    hourly_mean_10_day.timestamp = offset_data_by_business_days(hourly_mean_10_day.timestamp, 1)
    hourly_mean_20_day['timestamp'] = hourly_mean_20_day.index
    hourly_mean_20_day.timestamp = offset_data_by_business_days(hourly_mean_20_day.timestamp, 1)

    hourly_mean_data = merge_data(df, hourly_mean, 'close_hourly_mean', [hourly_mean.index.date, hourly_mean.index.hour], [df.index.date, df.index.hour])
    hourly_mean_10_day_data = merge_data(df, hourly_mean_10_day, 'close_hourly_mean', [hourly_mean_10_day.index.date, hourly_mean_10_day.index.hour], [df.index.date, df.index.hour])
    hourly_mean_20_day_data = merge_data(df, hourly_mean_20_day, 'close_hourly_mean', [hourly_mean_20_day.index.date, hourly_mean_20_day.index.hour], [df.index.date, df.index.hour])

    # Get Buy and Sell signals for 1-day hourly mean
    hourly_mean_data = get_buy_and_sell_signals(hourly_mean_data, 'close', 'close_hourly_mean')
    hourly_mean_20_day_data['10_day'] = hourly_mean_10_day_data['close_hourly_mean']
    crossover_hourly = get_buy_and_sell_signals(hourly_mean_20_day_data, 'close_hourly_mean', '10_day')

    df['pct_change'] = df['close'].pct_change(freq=5*pd.tseries.offsets.Minute())
    pct_change_data = get_buy_and_sell_signals_ROC(df[df['pct_change'].notna()], 'pct_change')

    # Combine SMA and ROC to get Momentum trading signal
    pct_change_data.rename(columns={'buy':'buy_roc'}, inplace=True)
    pct_change_data.rename(columns={'sell':'sell_roc'}, inplace=True)
    combined = merge_data(pct_change_data, crossover, ['buy', 'sell'], crossover.timestamp, pct_change_data.timestamp)
    combined_data = get_buy_and_sell_signals_combined(combined)


    # Create plot showing all methods
    plot(x_axis=one_day_sma_data['timestamp'],
        y_axis=[one_day_sma_data['close'], baseline_buy_signal['close'], baseline_sell_signal['close'], one_day_sma_data['buy'],
        one_day_sma_data['sell'], hourly_mean_data['buy'], hourly_mean_data['sell'], pct_change_data[~pct_change_data['buy_roc'].isna()]['close'], pct_change_data[~pct_change_data['sell_roc'].isna()]['close']],
        x_axis_name='Timestamp (EST)',
        y_axis_name='Close Price (USD)',
        title=f"Stock: {stock_symbol} | {year} | Rule Based Methods",
        x_axis_format="date",
        legend_labels=[None, "Baseline Buy", "Baseline Sell", "Daily SMA Buy", "Daily SMA Sell", "Hourly SMA Buy", "Hourly SMA Sell", "Momentum Buy", "Momentum Sell"],
        colors=['#bcbcbc', '#9fc5e8', '#0b5394', '#93c47d', '#38761d', '#ffd966', '#bf9000', '#ea9999', '#cc0000'],
        marker=[None, '^', 'v', '^', 'v', '^', 'v', '^', 'v'],
        linestyle=['-', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None'],
        markersize=2,
        alpha=[None, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        file_name=f"{dir_name}/{stock_symbol}_{year}.svg",
        show_plot=False
        )

    # Compute profits from all methods

    baseline_buy_signal.drop('timestamp', axis=1, inplace=True)
    baseline_sell_signal.drop('timestamp', axis=1, inplace=True)
    baseline_buy_signal.reset_index(inplace=True)
    baseline_sell_signal.reset_index(inplace=True)
    total_profits_baseline, percent_change_baseline = determine_profits(baseline_buy_signal["close"], baseline_sell_signal["close"])

    trade_signals = one_day_sma_data[one_day_sma_data['buy'].notna() | one_day_sma_data['sell'].notna()][['buy', 'sell']]
    total_profits_sma1, percent_change_sma1 = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    trade_signals = two_day_sma_data[two_day_sma_data['buy'].notna() | two_day_sma_data['sell'].notna()][['buy', 'sell']]
    total_profits_sma2, percent_change_sma2 = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    trade_signals = three_day_sma_data[three_day_sma_data['buy'].notna() | three_day_sma_data['sell'].notna()][['buy', 'sell']]
    total_profits_sma3, percent_change_sma3 = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    trade_signals = ten_day_sma_data[ten_day_sma_data['buy'].notna() | ten_day_sma_data['sell'].notna()][['buy', 'sell']]
    total_profits_sma10, percent_change_sma10 = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    trade_signals = crossover[crossover['buy'].notna() | crossover['sell'].notna()][['buy', 'sell']]
    total_profits_crossover, percent_change_crossover = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))


    trade_signals = hourly_mean_data[hourly_mean_data['buy'].notna() | hourly_mean_data['sell'].notna()][['buy', 'sell']]
    total_profits_hourly_mean, percent_change_hourly_mean = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    trade_signals = crossover_hourly[crossover_hourly['buy'].notna() | crossover_hourly['sell'].notna()][['buy', 'sell']]
    total_profits_crossover_hourly, percent_change_crossover_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))


    trade_signals = pct_change_data[pct_change_data['buy_roc'].notna() | pct_change_data['sell_roc'].notna()][['buy_roc', 'sell_roc']]
    total_profits_momentum, percent_change_momentum = determine_profits(trade_signals['buy_roc'], trade_signals['sell_roc'].shift(-1))

    trade_signals = combined_data[combined_data['buy'].notna() | combined_data['sell'].notna()][['buy', 'sell']]
    total_profits_momentum2, percent_change_momentum2 = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    total_buys_sma_crossover = get_total_trades(crossover[crossover['buy'].notna()]['buy'])
    total_sells_sma_crossover = get_total_trades(crossover[crossover['sell'].notna()]['sell'])
    # trades_per_hour = get_total_trades_per_hour(crossover[crossover['buy'].notna()], crossover[crossover['sell'].notna()])
    # trade_signals = crossover[crossover['buy'].notna() | crossover['sell'].notna()][['buy', 'sell']]
    # profits_per_hour = get_total_profits_per_hour(trade_signals['buy'], trade_signals['sell'].shift(-1))

    # Create table
    profit_values = {'Strategy' : ['Baseline', 'SMA-1', 'SMA-10', 'SMA Crossover', 'Hourly Mean', 'Hourly Crossover', 'Momentum', 'Momentum2'],
            'Total Profits (USD)': [total_profits_baseline, total_profits_sma1, total_profits_sma10, total_profits_crossover, total_profits_hourly_mean, total_profits_crossover_hourly, total_profits_momentum, total_profits_momentum2],
            'Percent Change (%)': [percent_change_baseline, percent_change_sma1, percent_change_sma10, percent_change_crossover, percent_change_hourly_mean, percent_change_crossover_hourly, percent_change_momentum, percent_change_momentum2]}
    profit_values_table = pd.DataFrame(profit_values)

    dfi.export(profit_values_table.style, f'{tables_dir_name}/{stock_symbol}_{year}_table.png')

    strategies = {'baseline': total_profits_baseline, 'sma_crossover': total_profits_crossover, 'hourly_crossover': total_profits_crossover_hourly}
    best_strategy = max(strategies, key=strategies.get)
    best_strategies[stock_symbol] = best_strategy
