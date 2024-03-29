from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import yaml
from datetime import datetime
import pytz
import pandas as pd
from determine_trade_times import get_baseline_signals, get_sma_crossover_signal, get_hourly_sma_crossover_signal, \
get_slow_stochastic_oscillator, get_hourly_slow_stochastic_oscillator, get_mean_reversion_signal, get_hourly_mean_reversion_signal, \
get_rsi_signal, get_hourly_rsi_signal
from analyze_trades import determine_profits, get_total_trades, get_total_trades_per_hour, get_total_profits_per_hour, get_total_returns_per_hour, format_trade_signals
import dataframe_image as dfi
from stock_split_info import adjust_for_stock_split
from plot_hourly_data import plot_hourly_profits, plot_hourly_number_of_trades, plot_hourly_returns
from plot_original_data import plot_original_data_year_with_trade_markers

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
data_path = params.get("data_path")
account_data_path = params.get("account_data_path")
dir_name = params.get("rule_based_data_plot_directory")
tables_dir_name = params.get("historical_directory")
stock_symbols = params.get("stock_symbols")
year = params.get("year")

start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST
est = pytz.timezone('US/Eastern')

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Get an entire year of data
data_start_date = datetime(year, 1, 1)
data_end_date = datetime(year, 12, 31)

methods = ['Baseline', 'SMA Crossover', 'Hourly Crossover', 'Slow Stoch Osc', 'Slow Stoch Hourly', 'Mean Reversion', 'Mean Reversion Hourly', 'RSI', 'RSI Hourly']

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

    # Baseline Buy and Sell signals- purely time based
    baseline_buy_signal, baseline_sell_signal = get_baseline_signals(df, tables_dir_name, 'svg')

    # SMA Crossover
    crossover_signal = get_sma_crossover_signal(df, data_start_date, 20, 50, tables_dir_name, 'svg')
    hourly_mean_crossover_signal = get_hourly_sma_crossover_signal(df, data_start_date, 20, 50, tables_dir_name, 'svg')

    # Slow Stochastic Oscillator
    slow_stochastic_oscillator = get_slow_stochastic_oscillator(df, data_start_date, '14D', '3D', 20, 80, tables_dir_name, 'svg')
    slow_stochastic_oscillator_hourly = get_hourly_slow_stochastic_oscillator(df, data_start_date, '14D', '3D', 20, 80, tables_dir_name, 'svg')

    # Mean Reversion Strategy
    mean_reversion_signal = get_mean_reversion_signal(df, data_start_date, '20D', [-1.5, 1.5], tables_dir_name, 'svg')
    mean_reversion_signal_hourly = get_hourly_mean_reversion_signal(df, data_start_date, '20D', [-1.5, 1.5], tables_dir_name, 'svg')

    # RSI
    rsi_signal = get_rsi_signal(df, data_start_date, '3D', 30, 70, dir_name=tables_dir_name, file_type='svg')
    rsi_signal_hourly = get_hourly_rsi_signal(df, data_start_date, '20D', 30, 70, dir_name=tables_dir_name, file_type='svg')

    y_axis=[df['close'], baseline_buy_signal['close'], baseline_sell_signal['close'], crossover_signal['buy'], crossover_signal['sell'], hourly_mean_crossover_signal['buy'], hourly_mean_crossover_signal['sell'],\
    slow_stochastic_oscillator['buy'], slow_stochastic_oscillator['sell'], slow_stochastic_oscillator_hourly['buy'], slow_stochastic_oscillator_hourly['sell'], mean_reversion_signal['buy'], mean_reversion_signal['sell'],\
    mean_reversion_signal_hourly['buy'], mean_reversion_signal_hourly['sell'], rsi_signal['buy'], rsi_signal['sell'], rsi_signal_hourly['buy'], rsi_signal_hourly['sell']]

    legend_labels=[None, "Baseline Buy", "Baseline Sell", "SMA Crossover Buy", "SMA Crossover Sell", "Hourly SMA Buy", "Hourly SMA Sell", \
    "Slow Stochastic Oscillator Buy", "Slow Stochastic Oscillator Sell", "Hourly Slow Stochastic Oscillator Buy", "Hourly Slow Stochastic Oscillator Sell",\
    "Mean Reversion Buy", "Mean Reversion Sell", "Hourly Mean Reversion Buy", "Hourly Mean Reversion Sell", "RSI Buy", "RSI Sell", "Hourly RSI Buy", "Hourly RSI Sell"]

    # Create plot showing all methods
    plot_original_data_year_with_trade_markers(stock_symbol, year, df['timestamp'], y_axis, legend_labels, tables_dir_name, 'svg')

    hourly_profits = []
    hourly_returns = []
    hourly_total_trades = []

    # Compute profits from all methods
    baseline_buy_signal = format_trade_signals(baseline_buy_signal, True)
    baseline_sell_signal = format_trade_signals(baseline_sell_signal, True)
    total_profits_baseline, percent_change_baseline = determine_profits(baseline_buy_signal["close"], baseline_sell_signal["close"])
    trade_counts = get_total_trades_per_hour(baseline_buy_signal, baseline_sell_signal, True)
    hourly_profits.append(get_total_profits_per_hour(baseline_buy_signal, baseline_sell_signal, True))
    hourly_returns.append(get_total_returns_per_hour(baseline_buy_signal, baseline_sell_signal, True))
    # plot_hourly_number_of_trades(buy_counts, sell_counts, 'Baseline', year, dir_name, 'png')

    trade_signals = format_trade_signals(crossover_signal)
    total_profits_crossover, percent_change_crossover = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    hourly_profits.append(get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))
    hourly_total_trades.append(get_total_trades_per_hour(trade_signals))
    hourly_returns.append(get_total_returns_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))

    trade_signals = format_trade_signals(hourly_mean_crossover_signal)
    total_profits_crossover_hourly, percent_change_crossover_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    hourly_profits.append(get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))
    hourly_total_trades.append(get_total_trades_per_hour(trade_signals))
    hourly_returns.append(get_total_returns_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))

    trade_signals = format_trade_signals(slow_stochastic_oscillator)
    total_profits_stoch, percent_change_stoch = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    hourly_profits.append(get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))
    hourly_total_trades.append(get_total_trades_per_hour(trade_signals))
    hourly_returns.append(get_total_returns_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))

    trade_signals = format_trade_signals(slow_stochastic_oscillator_hourly)
    total_profits_stoch_hourly, percent_change_stoch_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    hourly_profits.append(get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))
    hourly_total_trades.append(get_total_trades_per_hour(trade_signals))
    hourly_returns.append(get_total_returns_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))

    trade_signals = format_trade_signals(mean_reversion_signal)
    total_profits_mean_reversion, percent_change_mean_reversion = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    hourly_profits.append(get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))
    hourly_total_trades.append(get_total_trades_per_hour(trade_signals))
    hourly_returns.append(get_total_returns_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))

    trade_signals = format_trade_signals(mean_reversion_signal_hourly)
    total_profits_mean_reversion_hourly, percent_change_mean_reversion_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    hourly_profits.append(get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))
    hourly_total_trades.append(get_total_trades_per_hour(trade_signals))
    hourly_returns.append(get_total_returns_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))

    trade_signals = format_trade_signals(rsi_signal)
    total_profits_rsi, percent_change_rsi = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    hourly_profits.append(get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))
    hourly_total_trades.append(get_total_trades_per_hour(trade_signals))
    hourly_returns.append(get_total_returns_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))

    trade_signals = format_trade_signals(rsi_signal_hourly)
    total_profits_rsi_hourly, percent_change_rsi_hourly = determine_profits(trade_signals['buy'], trade_signals['sell'].shift(-1))
    hourly_profits.append(get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))
    hourly_total_trades.append(get_total_trades_per_hour(trade_signals))
    hourly_returns.append(get_total_returns_per_hour(trade_signals['buy'].shift(1), trade_signals['sell']))

    plot_hourly_profits(stock_symbol, hourly_profits, methods, year, tables_dir_name, 'svg')
    plot_hourly_returns(stock_symbol, hourly_returns, methods, year, tables_dir_name, 'svg')
    plot_hourly_number_of_trades(stock_symbol, hourly_total_trades, methods[1:], year, tables_dir_name, 'svg')

    # Create table
    profit_values = {'Strategy' : methods,
            'Total Profits (USD)': [total_profits_baseline, total_profits_crossover, total_profits_crossover_hourly, total_profits_stoch, total_profits_stoch_hourly, total_profits_mean_reversion, total_profits_mean_reversion_hourly, total_profits_rsi, total_profits_rsi_hourly],
            'Returns (%)': [percent_change_baseline, percent_change_crossover, percent_change_crossover_hourly, percent_change_stoch, percent_change_stoch_hourly, percent_change_mean_reversion, percent_change_mean_reversion_hourly, percent_change_rsi, percent_change_rsi_hourly]}
    profit_values_table = pd.DataFrame(profit_values)

    with open(f'{tables_dir_name}/{stock_symbol}_{year}_table.tex','w') as tf:
        tf.write(profit_values_table.to_latex())
    dfi.export(profit_values_table.style, f'{tables_dir_name}/{stock_symbol}_{year}_profit_table.svg')

    strategies = {'baseline': total_profits_baseline, 'sma_crossover': total_profits_crossover, 'hourly_crossover': total_profits_crossover_hourly, 'stoch_osc': total_profits_stoch, 'stoch_osc_hourly': total_profits_stoch_hourly, 'mean_reversion': total_profits_mean_reversion, 'mean_reversion_hourly': total_profits_mean_reversion_hourly, 'rsi': total_profits_rsi, 'rsi_hourly': total_profits_rsi_hourly}
    best_strategy = max(strategies, key=strategies.get)
    best_strategies[stock_symbol] = best_strategy

print(f"best strategies: {best_strategies}")
