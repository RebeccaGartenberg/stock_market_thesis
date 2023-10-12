import numpy as np

# https://www.youtube.com/watch?v=PUk5E8G1r44
def get_buy_and_sell_signals(data, col1, col2):
    data['signal'] = np.where(data[col1] < data[col2], 1, 0)
    data['position'] = data['signal'].diff()
    data['buy'] = np.where(data['position'] == 1, data[col1], np.NAN) # buy when current price falls below 1-day SMA
    data['sell'] = np.where(data['position'] == -1, data[col1], np.NAN) # sell when current price rises above 1-day SMA

    return data

def get_buy_and_sell_signals_ROC(data, col1):
    data['signal'] = np.where(data[col1] > 0, 1, np.where(data[col1] < 0, 0, float("nan"))) # change sign directions?
    data['position'] = data[data['signal'].notna()]['signal'].diff()
    data['buy'] = np.where(data['position'] == 1, data['close'], np.NAN) # buy when ROC is above 0
    data['sell'] = np.where(data['position'] == -1, data['close'], np.NAN) # sell when ROC is below 0

    return data

def get_buy_and_sell_signals_combined(data):
    data['position'] = data[(data['buy_roc'].notna() & data['buy'].notna()) | (data['sell_roc'].notna() & data['sell'].notna())]['signal'].diff()
    data['buy'] = np.where(data['position'] == 1, data['close'], np.NAN) # buy when ROC is above 0
    data['sell'] = np.where(data['position'] == -1, data['close'], np.NAN) # sell when ROC is below 0

    return data

# https://stackoverflow.com/questions/52909610/pandas-getting-first-and-last-value-from-each-day-in-a-datetime-dataframe
def get_baseline_signals(data):
    baseline_buy_signal = data.groupby(data.index.date).apply(lambda x: x.iloc[[0]]) # first datapoint from each day
    baseline_sell_signal = data.groupby(data.index.date).apply(lambda x: x.iloc[[-1]]) # last datapoint from each day
    baseline_buy_signal.index = baseline_buy_signal.index.droplevel(0)
    baseline_sell_signal.index = baseline_sell_signal.index.droplevel(0)

    return baseline_buy_signal, baseline_sell_signal
