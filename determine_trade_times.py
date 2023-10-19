import numpy as np
from aggregate_data import merge_data, offset_data_by_business_days, get_aggregated_mean, get_aggregated_mean_hourly
import pdb

def sma(data, grouping, n_days):
    # Aggregate Data
    sma = get_aggregated_mean(data, grouping, n_days) # 1-day SMA
    # Offset data by 1 business day to compare each value to previous day's mean
    sma.timestamp = offset_data_by_business_days(sma.timestamp, 1)
    # Merge n-day SMA means with dataframe
    sma_data = merge_data(data, sma, 'close', sma.timestamp.dt.date, data.timestamp.dt.date, True, 'timestamp')
    # Get Buy and Sell signals for n-day SMA
    sma_data = get_buy_and_sell_signals(sma_data, 'close', 'close_mean')

    return sma_data

def hourly_sma(data, n_days):
    # Aggregate Data
    hourly_mean = get_aggregated_mean_hourly(data, n_days)
    # Offset data by 1 business day to compare each value to previous day's mean
    hourly_mean['timestamp'] = hourly_mean.index
    hourly_mean.timestamp = offset_data_by_business_days(hourly_mean.timestamp, 1)
    # Merge n-day SMA means with dataframe
    hourly_mean_data = merge_data(data, hourly_mean, 'close_hourly_mean', [hourly_mean.index.date, hourly_mean.index.hour], [data.index.date, data.index.hour])
    # Get Buy and Sell signals for n-day SMA
    hourly_mean_data = get_buy_and_sell_signals(hourly_mean_data, 'close', 'close_hourly_mean')

    return hourly_mean_data

def get_sma_crossover_signal(data, short_time_period, long_time_period):
    sma_short = sma(data, data.timestamp.dt.date, short_time_period)
    sma_long = sma(data, data.timestamp.dt.date, long_time_period)
    sma_long[f'sma_{short_time_period}_day'] = sma_short['close_mean']
    crossover_signal = get_buy_and_sell_signals(sma_long, 'close_mean', f'sma_{short_time_period}_day')

    return crossover_signal

def get_hourly_sma_crossover_signal(data, short_time_period, long_time_period):
    hourly_mean_short = hourly_sma(data, short_time_period)
    hourly_mean_long = hourly_sma(data, long_time_period)
    hourly_mean_long[f'hourly_mean_{short_time_period}_day'] = hourly_mean_short['close_hourly_mean']
    crossover_signal = get_buy_and_sell_signals(hourly_mean_long, 'close_hourly_mean', f'hourly_mean_{short_time_period}_day')
    crossover_signal.index = crossover_signal.timestamp

    return crossover_signal


def get_slow_stochastic_oscillator(data, k, d, low_thresh, high_thresh):
    new_data = data.copy(deep=True)
    # Calculate the %K line
    new_data['lowest_low'] = new_data['low'].rolling(k).min()
    new_data['highest_high'] = new_data['high'].rolling(k).max()
    new_data['%K'] = ((new_data['close'] - new_data['lowest_low']) / (new_data['highest_high'] - new_data['lowest_low'])) * 100
    # Calculate the %D line
    new_data['%D'] = new_data['%K'].rolling(d).mean()

    new_data['signal'] = np.where(((new_data['%D'] > low_thresh) & (new_data['%D'].shift(-1) <= low_thresh)), 1, np.where(((new_data['%D'] < high_thresh) & (new_data['%D'].shift(-1) >= high_thresh)), 0, float("nan")))
    new_data['position'] = new_data[new_data['signal'].notna()]['signal'].diff()
    new_data['buy'] = np.where(new_data['position'] == 1, new_data['close'], np.NAN) # buy when current price falls below 1-day SMA
    new_data['sell'] = np.where(new_data['position'] == -1, new_data['close'], np.NAN) # sell when current price rises above 1-day SMA
    new_data['low_thresh'] = low_thresh
    new_data['high_thresh'] = high_thresh

    return new_data

def get_hourly_slow_stochastic_oscillator(data, k, d, low_thresh, high_thresh):
    # Calculate the %K line
    # group data by day and hour
    hourly_low = get_aggregated_mean_hourly(data, k, 'min')
    hourly_high = get_aggregated_mean_hourly(data, k, 'max')
    hourly_low['timestamp'] = hourly_low.index
    hourly_high['timestamp'] = hourly_high.index

    hourly_low.timestamp = offset_data_by_business_days(hourly_low.timestamp, 1)
    hourly_high.timestamp = offset_data_by_business_days(hourly_high.timestamp, 1)

    # Merge n-day SMA means with dataframe
    hourly_data = merge_data(data, hourly_low, 'lowest_low', [hourly_low.index.date, hourly_low.index.hour], [data.index.date, data.index.hour])
    hourly_data.index = hourly_data['timestamp']
    hourly_data.drop(['key_0', 'key_1'], axis=1, inplace=True)
    hourly_data = merge_data(hourly_data, hourly_high, 'highest_high', [hourly_high.index.date, hourly_high.index.hour], [hourly_data.index.date, hourly_data.index.hour])
    hourly_data.index = hourly_data['timestamp']

    hourly_data['%K'] = ((hourly_data['close'] - hourly_data['lowest_low']) / (hourly_data['highest_high'] - hourly_data['lowest_low'])) * 100
    # # Calculate the %D line
    hourly_data['%D'] = hourly_data['%K'].rolling(d).mean()

    hourly_data['signal'] = np.where(((hourly_data['%D'] > low_thresh) & (hourly_data['%D'].shift(-1) <= low_thresh)), 1, np.where(((hourly_data['%D'] < high_thresh) & (hourly_data['%D'].shift(-1) >= high_thresh)), 0, float("nan")))
    hourly_data['position'] = hourly_data[hourly_data['signal'].notna()]['signal'].diff()
    hourly_data['buy'] = np.where(hourly_data['position'] == 1, hourly_data['close'], np.NAN) # buy when current price falls below 1-day SMA
    hourly_data['sell'] = np.where(hourly_data['position'] == -1, hourly_data['close'], np.NAN) # sell when current price rises above 1-day SMA
    hourly_data['low_thresh'] = low_thresh
    hourly_data['high_thresh'] = high_thresh

    return hourly_data

def get_mean_reversion_signal(data, n_days, threshold):
    new_data = data.copy(deep=True)
    # Calculate the %K line
    new_data['close_mean'] = new_data['close'].rolling(n_days).mean()
    new_data['deviation'] = new_data['close'] - new_data['close_mean']
    new_data['std_dev'] = (new_data['deviation'].pow(2)/new_data['close'].rolling(n_days).count()).pow(1/2)
    new_data['z_score'] = new_data['deviation']/new_data['std_dev']

    new_data['signal'] = (np.where((new_data['z_score'] < threshold[0]), 1,
                        np.where(new_data['z_score'] > threshold[1], 0, float("nan"))))
    new_data['position'] = new_data[new_data['signal'].notna()]['signal'].diff() #new_data['signal'].diff()
    new_data['buy'] = np.where(new_data['position'] == 1, new_data['close'], np.NAN)
    new_data['sell'] = np.where(new_data['position'] == -1, new_data['close'], np.NAN)

    return new_data

def get_hourly_mean_reversion_signal(data, n_days, threshold):
    hourly_mean = get_aggregated_mean_hourly(data, n_days, 'mean')
    hourly_counts = get_aggregated_mean_hourly(data, n_days, 'count')
    hourly_mean['timestamp'] = hourly_mean.index
    hourly_counts['timestamp'] = hourly_counts.index

    hourly_mean.timestamp = offset_data_by_business_days(hourly_mean.timestamp, 1)
    hourly_counts.timestamp = offset_data_by_business_days(hourly_counts.timestamp, 1)

    hourly_data = merge_data(data, hourly_mean, 'close_hourly_mean', [hourly_mean.index.date, hourly_mean.index.hour], [data.index.date, data.index.hour])
    hourly_data.index = hourly_data['timestamp']
    hourly_data.drop(['key_0', 'key_1'], axis=1, inplace=True)
    hourly_data = merge_data(hourly_data, hourly_counts, 'count', [hourly_counts.index.date, hourly_counts.index.hour], [hourly_data.index.date, hourly_data.index.hour])
    hourly_data.index = hourly_data['timestamp']

    hourly_data['deviation'] = hourly_data['close'] - hourly_data['close_hourly_mean']
    hourly_data['std_dev'] = (hourly_data['deviation'].pow(2)/hourly_data['count']).pow(1/2)
    hourly_data['z_score'] = hourly_data['deviation']/hourly_data['std_dev']

    hourly_data['signal'] = (np.where((hourly_data['z_score'] < threshold[0]), 1,
                        np.where(hourly_data['z_score'] > threshold[1], 0, float("nan"))))
    hourly_data['position'] = hourly_data[hourly_data['signal'].notna()]['signal'].diff() #hourly_data['signal'].diff()
    hourly_data['buy'] = np.where(hourly_data['position'] == 1, hourly_data['close'], np.NAN)
    hourly_data['sell'] = np.where(hourly_data['position'] == -1, hourly_data['close'], np.NAN)

    return hourly_data


def get_rsi_signal(data, n_days, low_thresh, high_thresh, col='close'):
    new_data = data.copy(deep=True)
    new_data['rsi'] = calculate_rsi(data, col, n_days)
    rsi_upward = (new_data['rsi'] > low_thresh) & (new_data['rsi'].shift(-1) <= low_thresh)
    rsi_downward = (new_data['rsi'] < high_thresh) & (new_data['rsi'].shift(-1) >= high_thresh)

    new_data['signal'] = np.where(rsi_upward, 1, np.where(rsi_downward, 0, float("nan")))
    new_data['position'] = new_data[new_data['signal'].notna()]['signal'].diff() #new_data['signal'].diff()
    new_data['buy'] = np.where(new_data['position'] == 1, new_data['close'], np.NAN)
    new_data['sell'] = np.where(new_data['position'] == -1, new_data['close'], np.NAN)

    return new_data

def get_hourly_rsi_signal(data, n_days, low_thresh, high_thresh, col='close'):

    price_differences = data[col].diff()
    data['gain'] = price_differences.where(price_differences > 0, 0)
    data['loss'] = -price_differences.where(price_differences < 0, 0)

    hourly_gain = get_aggregated_mean_hourly(data, n_days, 'mean', 'gain')
    hourly_loss = get_aggregated_mean_hourly(data, n_days, 'mean', 'loss')

    hourly_gain['timestamp'] = hourly_gain.index
    hourly_loss['timestamp'] = hourly_loss.index

    hourly_gain.timestamp = offset_data_by_business_days(hourly_gain.timestamp, 1)
    hourly_loss.timestamp = offset_data_by_business_days(hourly_loss.timestamp, 1)

    hourly_data = merge_data(data, hourly_gain, 'avg_gain', [hourly_gain.index.date, hourly_gain.index.hour], [data.index.date, data.index.hour])
    hourly_data.index = hourly_data['timestamp']
    hourly_data.drop(['key_0', 'key_1'], axis=1, inplace=True)
    hourly_data = merge_data(hourly_data, hourly_loss, 'avg_loss', [hourly_loss.index.date, hourly_loss.index.hour], [hourly_data.index.date, hourly_data.index.hour])
    hourly_data.index = hourly_data['timestamp']

    rs = hourly_data['avg_gain'] / hourly_data['avg_loss']
    hourly_data['rsi'] = 100 - (100 / (1 + rs))

    rsi_upward = (hourly_data['rsi'] > low_thresh) & (hourly_data['rsi'].shift(-1) <= low_thresh)
    rsi_downward = (hourly_data['rsi'] < high_thresh) & (hourly_data['rsi'].shift(-1) >= high_thresh)

    hourly_data['signal'] = np.where(rsi_upward, 1, np.where(rsi_downward, 0, float("nan")))
    hourly_data['position'] = hourly_data[hourly_data['signal'].notna()]['signal'].diff() #hourly_data['signal'].diff()
    hourly_data['buy'] = np.where(hourly_data['position'] == 1, hourly_data['close'], np.NAN)
    hourly_data['sell'] = np.where(hourly_data['position'] == -1, hourly_data['close'], np.NAN)

    return hourly_data

def calculate_rsi(data, col, period):
    price_differences = data[col].diff()
    gain = price_differences.where(price_differences > 0, 0)
    loss = -price_differences.where(price_differences < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# https://www.youtube.com/watch?v=PUk5E8G1r44
def get_buy_and_sell_signals(data, col1, col2, col3='close'):
    data['signal'] = np.where(data[col2] > data[col1], 1, 0)
    data['position'] = data['signal'].diff()
    data['buy'] = np.where(data['position'] == 1, data[col3], np.NAN) # buy when current price falls below 1-day SMA
    data['sell'] = np.where(data['position'] == -1, data[col3], np.NAN) # sell when current price rises above 1-day SMA

    return data

# https://stackoverflow.com/questions/52909610/pandas-getting-first-and-last-value-from-each-day-in-a-datetime-dataframe
def get_baseline_signals(data):
    baseline_buy_signal = data.groupby(data.index.date).apply(lambda x: x.iloc[[0]]) # first datapoint from each day
    baseline_sell_signal = data.groupby(data.index.date).apply(lambda x: x.iloc[[-1]]) # last datapoint from each day
    baseline_buy_signal.index = baseline_buy_signal.index.droplevel(0)
    baseline_sell_signal.index = baseline_sell_signal.index.droplevel(0)

    return baseline_buy_signal, baseline_sell_signal
