from alpaca.data.requests import StockLatestQuoteRequest, StockQuotesRequest, StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import yaml
import pdb
from datetime import datetime, timezone, date, timedelta
import pytz
import matplotlib.pyplot as plt
import pandas as pd
from determine_trade_times import get_buy_and_sell_signals, get_baseline_signals, get_sma_crossover_signal, get_hourly_sma_crossover_signal, \
get_slow_stochastic_oscillator, get_hourly_slow_stochastic_oscillator, get_mean_reversion_signal, get_hourly_mean_reversion_signal, \
get_rsi_signal, get_hourly_rsi_signal
from aggregate_data import merge_data, offset_data_by_business_days, get_aggregated_mean, get_aggregated_mean_hourly
from plot_stock_data import plot
from analyze_trades import determine_profits, get_total_trades, get_total_trades_per_hour, get_total_profits_per_hour, format_trade_signals
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

    # Get SMA crossover signal of 10 and 20 day moving averages
    crossover_signal = get_sma_crossover_signal(df, 20, 50)

    # Get Hourly SMA crossover signal of 10 and 20 day hourly moving averages
    hourly_mean_crossover_signal = get_hourly_sma_crossover_signal(df, 10, 20)

    slow_stochastic_oscillator = get_slow_stochastic_oscillator(df, '14D', '3D', 20, 80)

    slow_stochastic_oscillator_hourly = get_hourly_slow_stochastic_oscillator(df, '14D', '3D', 20, 80)

    # Mean Reversion Strategy
    mean_reversion_signal = get_mean_reversion_signal(df, '20D', [-1.5, 1.5])

    mean_reversion_signal_hourly = get_hourly_mean_reversion_signal(df, '20D', [-1.5, 1.5])

    # RSI
    rsi_signal = get_rsi_signal(df, '20D', 30, 70)
    rsi_signal_hourly = get_hourly_rsi_signal(df, '20D', 30, 70)

    # Create plot showing all methods
    plot(x_axis=df['timestamp'],
        y_axis=[df['close'], baseline_buy_signal['close'], baseline_sell_signal['close'], crossover_signal['buy'],
        crossover_signal['sell'], hourly_mean_crossover_signal['buy'], hourly_mean_crossover_signal['sell']],
        x_axis_name='Timestamp (EST)',
        y_axis_name='Close Price (USD)',
        title=f"Stock: {stock_symbol} | {year} | Rule Based Methods",
        x_axis_format="date",
        legend_labels=[None, "Baseline Buy", "Baseline Sell", "SMA Crossover Buy", "SMA Crossover Sell", "Hourly SMA Buy", "Hourly SMA Sell"],
        colors=['#bcbcbc', '#9fc5e8', '#0b5394', '#93c47d', '#38761d', '#ffd966', '#bf9000'],
        marker=[None, '^', 'v', '^', 'v', '^', 'v'],
        linestyle=['-', 'None', 'None', 'None', 'None', 'None', 'None'],
        markersize=2,
        alpha=[None, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        file_name=f"{dir_name}/{stock_symbol}_{year}.svg",
        show_plot=False
        )

    # Compute profits from all methods
    baseline_buy_signal = format_trade_signals(baseline_buy_signal, True)
    baseline_sell_signal = format_trade_signals(baseline_sell_signal, True)
    total_profits_baseline, percent_change_baseline = determine_profits(baseline_buy_signal["close"], baseline_sell_signal["close"])

    trade_signals = format_trade_signals(crossover_signal)
    total_profits_crossover, percent_change_crossover = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    trade_signals = format_trade_signals(hourly_mean_crossover_signal)
    total_profits_crossover_hourly, percent_change_crossover_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    trade_signals = format_trade_signals(slow_stochastic_oscillator)
    total_profits_stoch, percent_change_stoch = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    trade_signals = format_trade_signals(slow_stochastic_oscillator_hourly)
    total_profits_stoch_hourly, percent_change_stoch_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    trade_signals = format_trade_signals(mean_reversion_signal)
    total_profits_mean_reversion, percent_change_mean_reversion = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    trade_signals = format_trade_signals(mean_reversion_signal_hourly)
    total_profits_mean_reversion_hourly, percent_change_mean_reversion_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    trade_signals = format_trade_signals(rsi_signal)
    total_profits_rsi, percent_change_rsi = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    trade_signals = format_trade_signals(rsi_signal_hourly)
    total_profits_rsi_hourly, percent_change_rsi_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))

    # Create table
    profit_values = {'Strategy' : ['Baseline', 'SMA Crossover', 'Hourly Crossover', 'Slow Stoch Osc', 'Slow Stoch Hourly', 'Mean Reversion', 'Mean Reversion Hourly', 'RSI', 'RSI Hourly'],
            'Total Profits (USD)': [total_profits_baseline, total_profits_crossover, total_profits_crossover_hourly, total_profits_stoch, total_profits_stoch_hourly, total_profits_mean_reversion, total_profits_mean_reversion_hourly, total_profits_rsi, total_profits_rsi_hourly],
            'Percent Change (%)': [percent_change_baseline, percent_change_crossover, percent_change_crossover_hourly, percent_change_stoch, percent_change_stoch_hourly, percent_change_mean_reversion, percent_change_mean_reversion_hourly, percent_change_rsi, percent_change_rsi_hourly]}
    profit_values_table = pd.DataFrame(profit_values)

    dfi.export(profit_values_table.style, f'{tables_dir_name}/{stock_symbol}_{year}_table.png')

    strategies = {'baseline': total_profits_baseline, 'sma_crossover': total_profits_crossover, 'hourly_crossover': total_profits_crossover_hourly, 'stoch_osc': total_profits_stoch, 'stoch_osc_hourly': total_profits_stoch_hourly, 'mean_reversion': total_profits_mean_reversion, 'mean_reversion_hourly': total_profits_mean_reversion_hourly, 'rsi': total_profits_rsi, 'rsi_hourly': total_profits_rsi_hourly}
    best_strategy = max(strategies, key=strategies.get)
    best_strategies[stock_symbol] = best_strategy

print(f"best strategies: {best_strategies}")
