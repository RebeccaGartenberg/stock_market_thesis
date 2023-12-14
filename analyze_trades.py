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
    # percent_change = (total_profits/buy_signal.sum()) * 100
    percent_change = (total_profits / buy_signal.max()) * 100
    # print(f'total profits: {total_profits}')
    # print(f'max buy price: {buy_signal.max()}')
    # percent_change = (price_differences / buy_signal)
    # total_percent_change = (percent_change.sum() + percent_change.prod()) * 100 # or just percent_change.prod() ?
    # compound_pct_change = (1 + (percent_change / 100))
    # total_percent_change = (compound_pct_change.prod() - 1) * 100

    return total_profits, percent_change

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

    # plt.bar(buy_counts.index, buy_counts.values, width=0.4, label='Buys', align='center', alpha=0.7)
    # plt.bar(sell_counts.index+0.4, sell_counts.values, width=0.4, label='Sells', align='center', alpha=0.7)
    # plt.xlabel('Hour (EST)')
    # plt.ylabel('Number of Trades')
    # plt.title('Trades per Hour {method} {year}')
    # plt.legend(loc='best')
    # hour_labels = [f"{hour % 12} AM" if hour < 12 else f"{hour % 12 if hour > 12 else hour} PM" for hour in range(9, 18)]
    # plt.xticks(range(9, 18), hour_labels)
    # # Save or show figure
    # plt.show()

    return buy_counts, sell_counts

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

    # plt.bar(profits.index, profits.values)
    # plt.xlabel('Hour (EST)')
    # plt.ylabel('Profits per hour (USD)')
    # plt.title('Profits per Hour {method} {year}')
    # plt.xticks(range(9, 17))  # Set x-axis ticks to represent hours from 0 to 23
    # Save or show figure
    # plt.show()

    return profits
