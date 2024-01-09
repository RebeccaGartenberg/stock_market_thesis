import pdb
import pandas as pd
import matplotlib.pyplot as plt
import pytz

def format_trade_signals(data, baseline=False):
    if baseline:
        data.drop('timestamp', axis=1, inplace=True)
        data.reset_index(inplace=True)
        return data

    return data[data['buy'].notna() | data['sell'].notna()][['buy', 'sell']]

def determine_profits(buy_signal, sell_signal):
    price_differences = sell_signal - buy_signal
    total_profits = price_differences[price_differences.notna()].sum()

    rate_of_return = (1 + (price_differences / buy_signal)).prod() - 1

    # percent_change = (total_profits/buy_signal.sum()) * 100
    # percent_change = (total_profits / buy_signal.max()) * 100 # subtract money lost from buy signal max?
    # print(f'total profits: {total_profits}')
    # print(f'max buy price: {buy_signal.max()}')
    # percent_change = (price_differences / buy_signal)
    # total_percent_change = (percent_change.sum() + percent_change.prod()) * 100 # or just percent_change.prod() ?
    # compound_pct_change = (1 + (percent_change / 100))
    # total_percent_change = (compound_pct_change.prod() - 1) * 100

    return total_profits, rate_of_return

def get_total_trades(trade_signal, baseline=False):
    if baseline:
        return trade_signal['close'].count()
    return trade_signal['buy'].count() + trade_signal['sell'].count()

def get_total_trades_per_hour(trade_signal, trade_signal_2=None, baseline=False):
    est = pytz.timezone('US/Eastern')
    if baseline:
        trade_signal.index = trade_signal['timestamp']
        trade_signal = trade_signal.drop('timestamp', axis=1)
        trade_signal.index = pd.to_datetime(trade_signal.index).tz_convert(est)
        trade_signal_2.index = trade_signal_2['timestamp']
        trade_signal_2 = trade_signal_2.drop('timestamp', axis=1)
        trade_signal_2.index = pd.to_datetime(trade_signal_2.index).tz_convert(est)
        test = pd.concat([trade_signal, trade_signal_2])
        return test.groupby(test.index.hour)['close'].count()

    trade_signal.index =  pd.to_datetime(trade_signal.index).tz_convert(est)
    buy_counts = trade_signal.groupby(trade_signal.index.hour)['buy'].count()
    sell_counts = trade_signal.groupby(trade_signal.index.hour)['sell'].count()

    return buy_counts, sell_counts

def get_total_returns_per_hour(buy_signal, sell_signal, baseline=False):
    est = pytz.timezone('US/Eastern')
    if baseline:
        signal = sell_signal.merge(buy_signal['close'], left_index=True, right_index=True)
        signal['price_dif'] = signal['close_x'] - signal['close_y']
        signal.index = signal['timestamp']
        signal.index = pd.to_datetime(signal.index).tz_convert(est)
        signal['return_ratios'] = 1 + (signal['price_dif'] / signal['close_y'])
        returns = signal['return_ratios'].groupby(signal['return_ratios'].index.hour).prod() - 1
        return returns

    price_differences = sell_signal - buy_signal
    return_ratios = 1 + (price_differences / buy_signal)
    return_ratios.index =  pd.to_datetime(return_ratios.index).tz_convert(est)
    rate_of_return = return_ratios.groupby(return_ratios.index.hour).prod() - 1

    return rate_of_return

def get_total_profits_per_hour(buy_signal, sell_signal, baseline=False):
    est = pytz.timezone('US/Eastern')
    if baseline:
        signal = sell_signal.merge(buy_signal['close'], left_index=True, right_index=True)
        signal['price_dif'] = signal['close_x'] - signal['close_y']
        signal.index = signal['timestamp']
        signal.index =  pd.to_datetime(signal.index).tz_convert(est)
        profits = signal.groupby(signal.index.hour).sum()
        return profits['price_dif']

    price_differences = sell_signal - buy_signal
    price_differences.index =  pd.to_datetime(price_differences.index).tz_convert(est)
    profits = price_differences.groupby(price_differences.index.hour).sum()

    return profits
