from alpaca.data.requests import StockLatestQuoteRequest, StockQuotesRequest, StockBarsRequest
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
import random
import csv
import numpy as np


def get_stock_symbols(n):
    symbol_df = pd.read_csv('./stock_symbols.csv', error_bad_lines=False)
    symbols = symbol_df.index.values
    random.shuffle(symbols)
    if n > len(symbols):
        n = len(symbols)

    return symbols[0:n]

def generate_and_save_training_data(stock_symbols, data_client, start_date, end_date, file_name_1, file_name_2, col_names_hourly):
    start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
    end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST
    est = pytz.timezone('US/Eastern')
    year = start_date.year
    existing_symbols = pd.read_csv(file_name_1).symbol.values

    profitable_strategies = {}
    best_strategies = {}

    for stock_symbol in stock_symbols:
        if stock_symbol in existing_symbols:
            continue

        is_profitable = []
        total_profits = []
        total_trades = []

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

        df = adjust_for_stock_split(df, stock_symbol, year)

        hourly_df = pd.DataFrame()
        hourly_df.index = df.groupby(df.index.tz_convert(est).hour).all().index
        hourly_df['symbol'] = [stock_symbol] * len(hourly_df.index)
        hourly_df['hour'] = hourly_df.index

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

        # Compute profits from all methods
        baseline_buy_signal = format_trade_signals(baseline_buy_signal, True)
        baseline_sell_signal = format_trade_signals(baseline_sell_signal, True)
        total_profits_baseline, percent_change_baseline = determine_profits(baseline_buy_signal["close"], baseline_sell_signal["close"])
        is_profitable.append(int(total_profits_baseline > 0))
        total_profits.append(total_profits_baseline)
        total_trades.append(get_total_trades(baseline_buy_signal, True) + get_total_trades(baseline_sell_signal, True))
        # buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        # hourly_df['sma_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(baseline_buy_signal, baseline_sell_signal, True)
        hourly_df['baseline_profits'] = hourly_profits
        hourly_df['baseline_is_prof'] = (hourly_profits > 0).astype(int)
        trade_counts = get_total_trades_per_hour(baseline_buy_signal, baseline_sell_signal, True)
        hourly_df['baseline_total_trades'] = trade_counts

        trade_signals = format_trade_signals(crossover_signal)
        total_profits_crossover, percent_change_crossover = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
        is_profitable.append(int(total_profits_crossover > 0))
        total_profits.append(total_profits_crossover)
        total_trades.append(get_total_trades(trade_signals))
        buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        hourly_df['sma_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
        hourly_df['sma_profits'] = hourly_profits
        hourly_df['sma_is_prof'] = (hourly_profits > 0).astype(int)

        trade_signals = format_trade_signals(hourly_mean_crossover_signal)
        total_profits_crossover_hourly, percent_change_crossover_hourly = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
        is_profitable.append(int(total_profits_crossover_hourly > 0))
        total_profits.append(total_profits_crossover_hourly)
        total_trades.append(get_total_trades(trade_signals))
        buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        hourly_df['sma_hourly_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
        hourly_df['sma_hourly_profits'] = hourly_profits
        hourly_df['sma_hourly_is_prof'] = (hourly_profits > 0).astype(int)

        trade_signals = format_trade_signals(slow_stochastic_oscillator)
        total_profits_stoch, percent_change_stoch = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
        is_profitable.append(int(total_profits_stoch > 0))
        total_profits.append(total_profits_stoch)
        total_trades.append(get_total_trades(trade_signals))
        buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        hourly_df['stoch_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
        hourly_df['stoch_profits'] = hourly_profits
        hourly_df['stoch_is_prof'] = (hourly_profits > 0).astype(int)

        trade_signals = format_trade_signals(slow_stochastic_oscillator_hourly)
        total_profits_stoch_hourly, percent_change_stoch_hourly = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
        is_profitable.append(int(total_profits_stoch_hourly > 0))
        total_profits.append(total_profits_stoch_hourly)
        total_trades.append(get_total_trades(trade_signals))
        buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        hourly_df['stoch_hourly_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
        hourly_df['stoch_hourly_profits'] = hourly_profits
        hourly_df['stoch_hourly_is_prof'] = (hourly_profits > 0).astype(int)

        trade_signals = format_trade_signals(mean_reversion_signal)
        total_profits_mean_reversion, percent_change_mean_reversion = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
        is_profitable.append(int(total_profits_mean_reversion > 0))
        total_profits.append(total_profits_mean_reversion)
        total_trades.append(get_total_trades(trade_signals))
        buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        hourly_df['mean_rever_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
        hourly_df['mean_rever_profits'] = hourly_profits
        hourly_df['mean_rever_is_prof'] = (hourly_profits > 0).astype(int)

        trade_signals = format_trade_signals(mean_reversion_signal_hourly)
        total_profits_mean_reversion_hourly, percent_change_mean_reversion_hourly = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
        is_profitable.append(int(total_profits_mean_reversion_hourly > 0))
        total_profits.append(total_profits_mean_reversion_hourly)
        total_trades.append(get_total_trades(trade_signals))
        buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        hourly_df['mean_rever_hourly_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
        hourly_df['mean_rever_hourly_profits'] = hourly_profits
        hourly_df['mean_rever_hourly_is_prof'] = (hourly_profits > 0).astype(int)

        trade_signals = format_trade_signals(rsi_signal)
        total_profits_rsi, percent_change_rsi = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
        is_profitable.append(int(total_profits_rsi > 0))
        total_profits.append(total_profits_rsi)
        total_trades.append(get_total_trades(trade_signals))
        buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        hourly_df['rsi_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
        hourly_df['rsi_profits'] = hourly_profits
        hourly_df['rsi_is_prof'] = (hourly_profits > 0).astype(int)

        trade_signals = format_trade_signals(rsi_signal_hourly)
        total_profits_rsi_hourly, percent_change_rsi_hourly = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
        is_profitable.append(int(total_profits_rsi_hourly > 0))
        total_profits.append(total_profits_rsi_hourly)
        total_trades.append(get_total_trades(trade_signals))
        buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
        hourly_df['rsi_hourly_total_trades'] = buy_counts + sell_counts
        hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
        hourly_df['rsi_hourly_profits'] = hourly_profits
        hourly_df['rsi_hourly_is_prof'] = (hourly_profits > 0).astype(int)

        profitable_strategies[stock_symbol] = is_profitable
        strategies = {'baseline': total_profits_baseline, 'sma_crossover': total_profits_crossover, 'hourly_crossover': total_profits_crossover_hourly, 'stoch_osc': total_profits_stoch, 'stoch_osc_hourly': total_profits_stoch_hourly, 'mean_reversion': total_profits_mean_reversion, 'mean_reversion_hourly': total_profits_mean_reversion_hourly, 'rsi': total_profits_rsi, 'rsi_hourly': total_profits_rsi_hourly}
        best_strategy = max(strategies, key=strategies.get)
        best_strategies[stock_symbol] = best_strategy

        # Summary Statistics
        mean_price = df.mean()['close']
        std_dev = df.std()['close']

        # log_returns =  np.log(df['close'].iloc[1:] / df['close'].iloc[1:].shift(1))
        # volatility = log_returns.std()

        # Hourly Summary Statistics
        hourly_df['mean_price'] = df.groupby(df.index.tz_convert(est).hour).mean()['close']
        hourly_df['std_dev'] = df.groupby(df.index.tz_convert(est).hour).std()['close']

        # get best strategy per hour
        cols_to_include = ['baseline_profits', 'sma_profits', 'sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits', 'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits']
        hourly_df['best_strategy'] = hourly_df[cols_to_include].idxmax(axis='columns')

        # save is_profitable, profits, total trades, best strategy
        with open(file_name_1, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([stock_symbol] + is_profitable + total_profits + total_trades + [mean_price] + [std_dev] + [best_strategy])

        hourly_df = hourly_df[col_names_hourly].fillna(0)
        hourly_df.to_csv(file_name_2, mode='a', header=False, index=False)

    return profitable_strategies, best_strategies
