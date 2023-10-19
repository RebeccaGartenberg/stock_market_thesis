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
    percent_change = (price_differences / buy_signal)
    total_percent_change = (percent_change.sum() + percent_change.prod()) * 100 # or just percent_change.prod() ?
    # compound_pct_change = (1 + (percent_change / 100))
    # total_percent_change = (compound_pct_change.prod() - 1) * 100

    return total_profits, total_percent_change

def get_total_trades(trade_signal):
    return trade_signal.count()

def get_total_trades_per_hour(buy_signal, sell_signal):
    est = pytz.timezone('US/Eastern')
    # buy_signal['timestamp'] = pd.to_datetime(buy_signal['timestamp']).dt.tz_convert(est)
    buy_signal.index =  pd.to_datetime(buy_signal.index).tz_convert(est)
    # buy_counts = buy_signal.groupby(buy_signal['timestamp'].dt.hour).count()['buy']
    buy_counts = buy_signal.groupby(buy_signal.index.hour).count()['buy']
    sell_signal.index = pd.to_datetime(sell_signal.index).tz_convert(est)
    sell_counts = sell_signal.groupby(sell_signal.index.hour).count()['sell']


    plt.bar(buy_counts.index, buy_counts.values, width=0.4, label='Buys', align='center', alpha=0.7)
    plt.bar(sell_counts.index+0.4, sell_counts.values, width=0.4, label='Sells', align='center', alpha=0.7)
    plt.xlabel('Hour (EST)')
    plt.ylabel('Number of Trades')
    plt.title('Trades per Hour {method} {year}')
    plt.legend(loc='best')
    hour_labels = [f"{hour % 12} AM" if hour < 12 else f"{hour % 12 if hour > 12 else hour} PM" for hour in range(9, 18)]
    plt.xticks(range(9, 18), hour_labels)
    # Save or show figure
    plt.show()

    return

def get_total_profits_per_hour(buy_signal, sell_signal):
    est = pytz.timezone('US/Eastern')

    price_differences = sell_signal - buy_signal
    price_differences.index =  pd.to_datetime(price_differences.index).tz_convert(est)
    profits = price_differences.groupby(price_differences.index.hour).sum()

    plt.bar(profits.index, profits.values)
    plt.xlabel('Hour (EST)')
    plt.ylabel('Profits per hour (USD)')
    plt.title('Profits per Hour {method} {year}')
    plt.xticks(range(9, 17))  # Set x-axis ticks to represent hours from 0 to 23
    # Save or show figure
    plt.show()

    return
