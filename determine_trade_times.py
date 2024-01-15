import numpy as np
from aggregate_data import merge_data, offset_data_by_business_days, get_aggregated_mean, get_aggregated_mean_hourly
import pdb
from datetime import timedelta
import pandas as pd
from plot_original_data import plot_original_data_trade_signals, plot_original_data_trade_signals_subplots

def sma_old(data, grouping, n_days):
    # Aggregate Data
    sma = get_aggregated_mean(data, grouping, n_days) # 1-day SMA
    # Offset data by 1 business day to compare each value to previous day's mean
    sma.timestamp = offset_data_by_business_days(sma.timestamp, 1)
    # Merge n-day SMA means with dataframe
    sma_data = merge_data(data, sma, 'close', sma.timestamp.dt.date, data.timestamp.dt.date, True, 'timestamp')
    # Get Buy and Sell signals for n-day SMA
    sma_data = get_buy_and_sell_signals(sma_data, 'close', 'close_mean')

    return sma_data

def sma(data, col, window):
    return data[col].rolling(window=window).mean()

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

# def hourly_sma_2(data, n_days):
#     # Aggregate Data
#     hourly_mean = get_aggregated_mean_hourly(data, n_days)
#     # Offset data by 1 business day to compare each value to previous day's mean
#     hourly_mean['timestamp'] = hourly_mean.index
#     hourly_mean.timestamp = offset_data_by_business_days(hourly_mean.timestamp, 1)
#     # Merge n-day SMA means with dataframe
#     hourly_mean_data = merge_data(data, hourly_mean, 'close_hourly_mean', [hourly_mean.index.date, hourly_mean.index.hour], [data.index.date, data.index.hour])
#     # Get Buy and Sell signals for n-day SMA
#     hourly_mean_data = get_buy_and_sell_signals(hourly_mean_data, 'close', 'close_hourly_mean')
#
#     return hourly_mean_data

def get_sma_crossover_signal_old(data, short_time_period, long_time_period):
    sma_short = sma_old(data, data.timestamp.dt.date, short_time_period)
    sma_long = sma_old(data, data.timestamp.dt.date, long_time_period)
    sma_long[f'sma_{short_time_period}_day'] = sma_short['close_mean']
    crossover_signal = get_buy_and_sell_signals(sma_long, 'close_mean', f'sma_{short_time_period}_day')

    return crossover_signal

def get_sma_crossover_signal(data, start_date, short_time_period, long_time_period, dir_name=None, file_type=None):
    sma_short = sma(data, 'close', f'{short_time_period}D')
    sma_long = sma(data, 'close', f'{long_time_period}D')
    sma_signal = pd.DataFrame()
    sma_signal['timestamp'] = data['timestamp']
    sma_signal['close'] = data['close']
    sma_signal[f'sma_{short_time_period}_day'] = sma_short
    sma_signal[f'sma_{long_time_period}_day'] = sma_long
    # filter out first x days to account for delay
    sma_signal = sma_signal[sma_signal['timestamp'].dt.date >= start_date.date()]
    # sma_signal = sma_signal[sma_signal['timestamp'] > sma_signal['timestamp'][0]+timedelta(long_time_period)]
    # crossover_signal = get_buy_and_sell_signals(sma_signal, f'sma_{long_time_period}_day', f'sma_{short_time_period}_day')
    crossover_signal = get_buy_and_sell_signals(sma_signal, f'sma_{short_time_period}_day', f'sma_{long_time_period}_day')

    if dir_name is not None:
        plot_original_data_trade_signals(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'],sma_short, sma_long],
                                        'sma_crossover',
                                        'SMA Crossover',
                                        ['Original data', f'{short_time_period} day SMA', f'{long_time_period} day SMA'],
                                        dir_name,
                                        file_type)

    return crossover_signal

def get_hourly_sma_crossover_signal_old(data, short_time_period, long_time_period):
    hourly_mean_short = hourly_sma(data, short_time_period)
    hourly_mean_long = hourly_sma(data, long_time_period)
    hourly_mean_long[f'hourly_mean_{short_time_period}_day'] = hourly_mean_short['close_hourly_mean']
    crossover_signal = get_buy_and_sell_signals(hourly_mean_long, 'close_hourly_mean', f'hourly_mean_{short_time_period}_day')
    crossover_signal.index = crossover_signal.timestamp

    return crossover_signal


def get_hourly_sma_crossover_signal(data, start_date, short_time_period, long_time_period, dir_name=None, file_type=None):

    hourly_sma_short = (
        data.groupby(data.index.hour)['close']
        .rolling(window=f'{short_time_period}D', min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        )

    hourly_sma_long = (
        data.groupby(data.index.hour)['close']
        .rolling(window=f'{long_time_period}D', min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        )

    hourly_sma_signal = pd.DataFrame()
    hourly_sma_signal['timestamp'] = data['timestamp']
    hourly_sma_signal['close'] = data['close']
    hourly_sma_signal[f'sma_{short_time_period}_day'] = hourly_sma_short
    hourly_sma_signal[f'sma_{long_time_period}_day'] = hourly_sma_long
    # filter out first x days to account for delay
    hourly_sma_signal = hourly_sma_signal[hourly_sma_signal['timestamp'].dt.date >= start_date.date()]
    # hourly_sma_signal = hourly_sma_signal[hourly_sma_signal['timestamp'] > hourly_sma_signal['timestamp'][0]+timedelta(long_time_period)]
    # hourly_crossover_signal = get_buy_and_sell_signals(hourly_sma_signal, f'sma_{long_time_period}_day', f'sma_{short_time_period}_day')
    hourly_crossover_signal = get_buy_and_sell_signals(hourly_sma_signal, f'sma_{short_time_period}_day', f'sma_{long_time_period}_day')

    if dir_name is not None:
        plot_original_data_trade_signals(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'], hourly_sma_signal[f'sma_{short_time_period}_day'], hourly_sma_signal[f'sma_{long_time_period}_day']],
                                        'hourly_sma_crossover',
                                        'Hourly SMA Crossover',
                                        ['Original data', f'Hourly {short_time_period} day SMA', f'Hourly {long_time_period} day SMA'],
                                        dir_name,
                                        file_type)

    return hourly_crossover_signal

def get_slow_stochastic_oscillator_old(data, k, d, low_thresh, high_thresh):
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

def get_slow_stochastic_oscillator(data, start_date, k, d, low_thresh, high_thresh, dir_name=None, file_type=None):
    stoch_osc = data.copy(deep=True)
    # Calculate the %K line
    stoch_osc['lowest_low'] = stoch_osc['low'].rolling(k).min()
    stoch_osc['highest_high'] = stoch_osc['high'].rolling(k).max()
    stoch_osc['%K'] = ((stoch_osc['close'] - stoch_osc['lowest_low']) / (stoch_osc['highest_high'] - stoch_osc['lowest_low'])) * 100
    # Calculate the %D line
    stoch_osc['%D'] = stoch_osc['%K'].rolling(d).mean()
    # filter out first x days to account for delay
    stoch_osc = stoch_osc[stoch_osc['timestamp'].dt.date >= start_date.date()]
    # hourly_osc = hourly_osc[hourly_osc['timestamp'] > hourly_osc['timestamp'][0]+timedelta(int(k.split('D')[0]))]

    if stoch_osc.empty:
        return stoch_osc

    # Combined
    stoch_osc.loc[(stoch_osc['%K'] < low_thresh) | (stoch_osc['%K'] > stoch_osc['%D']), 'signal'] = 1
    stoch_osc.loc[(stoch_osc['%K'] > high_thresh) | (stoch_osc['%K'] < stoch_osc['%D']), 'signal'] = 0

    stoch_osc['position'] = stoch_osc[stoch_osc['signal'].notna()]['signal'].diff()
    stoch_osc['buy'] = np.where(stoch_osc['position'] == 1, stoch_osc['close'], np.NAN) # buy when current price falls below 1-day SMA
    stoch_osc['sell'] = np.where(stoch_osc['position'] == -1, stoch_osc['close'], np.NAN) # sell when current price rises above 1-day SMA
    stoch_osc['low_thresh'] = low_thresh
    stoch_osc['high_thresh'] = high_thresh

    if dir_name is not None:
        plot_original_data_trade_signals_subplots(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'], stoch_osc['%K'], stoch_osc['%D']],
                                        'slow_stoch',
                                        'Slow Stochastic Oscillator',
                                        ['Original data', '%K line', '%D line'],
                                        dir_name,
                                        file_type
                                        )

    return stoch_osc

def get_hourly_slow_stochastic_oscillator_old(data, k, d, low_thresh, high_thresh):
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


def get_hourly_slow_stochastic_oscillator(data, start_date, k, d, low_thresh, high_thresh, dir_name=None, file_type=None):
    # Calculate the %K line
    # group data by day and hour
    hourly_low = (
        data.groupby(data.index.hour)['low']
        .rolling(window=k, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
        )

    hourly_high = (
        data.groupby(data.index.hour)['high']
        .rolling(window=k, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        )

    hourly_low['timestamp'] = hourly_low.index
    hourly_high['timestamp'] = hourly_high.index

    hourly_osc = pd.DataFrame()
    hourly_osc['timestamp'] = data['timestamp']
    hourly_osc['close'] = data['close']
    hourly_osc['lowest_low'] = hourly_low # maybe merge here based on timestamp
    hourly_osc['highest_high'] = hourly_high

    # filter out first x days to account for delay
    hourly_osc = hourly_osc[hourly_osc['timestamp'].dt.date >= start_date.date()]
    # hourly_osc = hourly_osc[hourly_osc['timestamp'] > hourly_osc['timestamp'][0]+timedelta(int(k.split('D')[0]))]

    # % K
    hourly_osc['%K'] = ((hourly_osc['close'] - hourly_osc['lowest_low']) / (hourly_osc['highest_high'] - hourly_osc['lowest_low'] + 1e-10)) * 100
    # # Calculate the %D line
    hourly_osc['%D'] = hourly_osc['%K'].rolling(d).mean()

    # Overbought / Oversold signals
    # hourly_osc.loc[hourly_osc['%K'] < low_thresh, 'signal'] = 1
    # hourly_osc.loc[hourly_osc['%K'] > high_thresh, 'signal'] = 0

    # Crossover Signals
    # signals.loc[data['%K'] > data['%D'], 'signal'] = 1
    # signals.loc[data['%K'] < data['%D'], 'signal'] = 0

    if hourly_osc.empty:
        return hourly_osc

    # Combined
    hourly_osc.loc[(hourly_osc['%K'] < low_thresh) | (hourly_osc['%K'] > hourly_osc['%D']), 'signal'] = 1
    hourly_osc.loc[(hourly_osc['%K'] > high_thresh) | (hourly_osc['%K'] < hourly_osc['%D']), 'signal'] = 0

    hourly_osc['position'] = hourly_osc[hourly_osc['signal'].notna()]['signal'].diff()
    hourly_osc['buy'] = np.where(hourly_osc['position'] == 1, hourly_osc['close'], np.NAN) # buy when current price falls below 1-day SMA
    hourly_osc['sell'] = np.where(hourly_osc['position'] == -1, hourly_osc['close'], np.NAN) # sell when current price rises above 1-day SMA
    hourly_osc['low_thresh'] = low_thresh
    hourly_osc['high_thresh'] = high_thresh

    if dir_name is not None:
        plot_original_data_trade_signals_subplots(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'], hourly_osc['%K'], hourly_osc['%D']],
                                        'hourly_slow_stoch',
                                        'Hourly Slow Stoc. Osc.',
                                        ['Original data', 'Hourly %K line', 'Hourly %D line'],
                                        dir_name,
                                        file_type)

    return hourly_osc

def get_mean_reversion_signal_old(data, n_days, threshold):
    new_data = data.copy(deep=True)
    # Calculate the %K line
    new_data['close_mean'] = new_data['close'].rolling(n_days).mean()
    new_data['deviation'] = new_data['close'] - new_data['close_mean']
    new_data['std_dev'] = (new_data['deviation'].pow(2)/new_data['close'].rolling(n_days).count()).pow(1/2)
    new_data['std_dev']  = new_data['std_dev'].replace(0, 1e-10)
    new_data['z_score'] = new_data['deviation']/new_data['std_dev']

    new_data['signal'] = (np.where((new_data['z_score'] < threshold[0]), 1,
                        np.where(new_data['z_score'] > threshold[1], 0, float("nan"))))
    new_data['position'] = new_data[new_data['signal'].notna()]['signal'].diff() #new_data['signal'].diff()
    new_data['buy'] = np.where(new_data['position'] == 1, new_data['close'], np.NAN)
    new_data['sell'] = np.where(new_data['position'] == -1, new_data['close'], np.NAN)

    return new_data

def get_mean_reversion_signal(data, start_date, n_days, threshold, dir_name=None, file_type=None):
    new_data = data.copy(deep=True)
    new_data['close_mean'] = new_data['close'].rolling(n_days).mean()
    new_data['deviation'] = new_data['close'] - new_data['close_mean']
    new_data['std_dev'] = (new_data['deviation'].pow(2)/new_data['close'].rolling(n_days).count()).pow(1/2)
    new_data['std_dev']  = new_data['std_dev'].replace(0, 1e-10)
    new_data['z_score'] = new_data['deviation']/new_data['std_dev']

    # filter out first x days to account for delay
    new_data = new_data[new_data['timestamp'].dt.date >= start_date.date()]
    # new_data = new_data[new_data['timestamp'] > new_data['timestamp'][0]+timedelta(int(n_days.split('D')[0]))]

    if new_data.empty:
        return new_data

    # Can add bolinger bands ***
    new_data.loc[new_data['z_score'] < threshold[0], 'signal'] = 1
    new_data.loc[new_data['z_score'] > threshold[1], 'signal'] = 0
    new_data['low_thresh'] = threshold[0]
    new_data['high_thresh'] = threshold[1]

    new_data['position'] = new_data[new_data['signal'].notna()]['signal'].diff() #new_data['signal'].diff()
    new_data['buy'] = np.where(new_data['position'] == 1, new_data['close'], np.NAN)
    new_data['sell'] = np.where(new_data['position'] == -1, new_data['close'], np.NAN)

    if dir_name is not None:
        plot_original_data_trade_signals_subplots(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'], new_data['z_score'], new_data['low_thresh'], new_data['high_thresh']],
                                        'mean_rever',
                                        'Mean Reversion',
                                        ['Original data', 'Z score', 'low threshold', 'high threshold'],
                                        dir_name,
                                        file_type)
    return new_data

def get_hourly_mean_reversion_signal_old(data, n_days, threshold):
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

def get_hourly_mean_reversion_signal(data, start_date, n_days, threshold, dir_name=None, file_type=None):

    hourly_mean = (
        data.groupby(data.index.hour)['close']
        .rolling(window=n_days, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        )

    hourly_std_dev = (
        data.groupby(data.index.hour)['close']
        .rolling(window=n_days, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        )

    hourly_mean['timestamp'] = hourly_mean.index
    hourly_std_dev['timestamp'] = hourly_std_dev.index
    hourly_data = pd.DataFrame()
    hourly_data['timestamp'] = data['timestamp']
    hourly_data['close'] = data['close']
    hourly_data['hourly_mean'] = hourly_mean # maybe merge here based on timestamp
    hourly_data['hourly_std_dev'] = hourly_std_dev
    hourly_data['deviation'] = hourly_data['close'] - hourly_data['hourly_mean']
    hourly_data['hourly_std_dev']  = hourly_data['hourly_std_dev'].replace(0, 1e-10)
    hourly_data['z_score'] = hourly_data['deviation']/hourly_data['hourly_std_dev']

    # filter out first x days to account for delay
    hourly_data = hourly_data[hourly_data['timestamp'].dt.date >= start_date.date()]
    # hourly_data = hourly_data[hourly_data['timestamp'] > hourly_data['timestamp'][0]+timedelta(int(n_days.split('D')[0]))]
    if hourly_data.empty:
        return hourly_data
    # Can add bolinger bands ***
    hourly_data.loc[hourly_data['z_score'] < threshold[0], 'signal'] = 1
    hourly_data.loc[hourly_data['z_score'] > threshold[1], 'signal'] = 0
    hourly_data['low_thresh'] = threshold[0]
    hourly_data['high_thresh'] = threshold[1]

    hourly_data['position'] = hourly_data[hourly_data['signal'].notna()]['signal'].diff() #hourly_data['signal'].diff()
    hourly_data['buy'] = np.where(hourly_data['position'] == 1, hourly_data['close'], np.NAN)
    hourly_data['sell'] = np.where(hourly_data['position'] == -1, hourly_data['close'], np.NAN)

    if dir_name is not None:
        plot_original_data_trade_signals_subplots(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'], hourly_data['z_score'], hourly_data['low_thresh'], hourly_data['high_thresh']],
                                        'hourly_mean_rever',
                                        'Hourly Mean Reversion',
                                        ['Original data', 'Hourly Z score', 'low threshold', 'high threshold'],
                                        dir_name,
                                        file_type)
    return hourly_data

def get_rsi_signal_old(data, n_days, low_thresh, high_thresh, col='close'):
    new_data = data.copy(deep=True)
    new_data['rsi'] = calculate_rsi(data, col, n_days)
    rsi_upward = (new_data['rsi'] > low_thresh) & (new_data['rsi'].shift(-1) <= low_thresh)
    rsi_downward = (new_data['rsi'] < high_thresh) & (new_data['rsi'].shift(-1) >= high_thresh)

    new_data['signal'] = np.where(rsi_upward, 1, np.where(rsi_downward, 0, float("nan")))
    new_data['position'] = new_data[new_data['signal'].notna()]['signal'].diff() #new_data['signal'].diff()
    new_data['buy'] = np.where(new_data['position'] == 1, new_data['close'], np.NAN)
    new_data['sell'] = np.where(new_data['position'] == -1, new_data['close'], np.NAN)

    return new_data


def get_rsi_signal(data, start_date, n_days, low_thresh, high_thresh, col='close', dir_name=None, file_type=None):
    new_data = data.copy(deep=True)
    new_data['rsi'] = calculate_rsi(data, col, n_days)

    if new_data.empty:
        return new_data
    new_data.loc[new_data['rsi'] < low_thresh, 'signal'] = 1
    new_data.loc[new_data['rsi'] > high_thresh, 'signal'] = 0
    new_data['low_thresh'] = low_thresh
    new_data['high_thresh'] = high_thresh

    # rsi_upward = (new_data['rsi'] > low_thresh) & (new_data['rsi'].shift(-1) <= low_thresh)
    # rsi_downward = (new_data['rsi'] < high_thresh) & (new_data['rsi'].shift(-1) >= high_thresh)

    # filter out first x days to account for delay
    new_data = new_data[new_data['timestamp'].dt.date >= start_date.date()]
    # new_data = new_data[new_data['timestamp'] > new_data['timestamp'][0]+timedelta(int(n_days.split('D')[0]))]

    # new_data['signal'] = np.where(rsi_upward, 1, np.where(rsi_downward, 0, float("nan")))
    new_data['position'] = new_data[new_data['signal'].notna()]['signal'].diff() #new_data['signal'].diff()
    new_data['buy'] = np.where(new_data['position'] == 1, new_data['close'], np.NAN)
    new_data['sell'] = np.where(new_data['position'] == -1, new_data['close'], np.NAN)

    if dir_name is not None:
        plot_original_data_trade_signals_subplots(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'], new_data['rsi'], new_data['low_thresh'], new_data['high_thresh']],
                                        'rsi',
                                        'RSI',
                                        ['Original data', 'RSI', 'low threshold', 'high threshold'],
                                        dir_name,
                                        file_type)
    return new_data

def get_hourly_rsi_signal_old(data, n_days, low_thresh, high_thresh, col='close'):
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

def get_hourly_rsi_signal(data, start_date, n_days, low_thresh, high_thresh, col='close', dir_name=None, file_type=None):
    price_differences = data[col].diff()
    data['gain'] = price_differences.where(price_differences > 0, 0)
    data['loss'] = abs(price_differences.where(price_differences < 0, 0))

    hourly_gain = (
        data.groupby(data.index.hour)['gain']
        .rolling(window=n_days, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        )

    hourly_loss = (
        data.groupby(data.index.hour)['loss']
        .rolling(window=n_days, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        )
    # hourly_gain = get_aggregated_mean_hourly(data, n_days, 'mean', 'gain')
    # hourly_loss = get_aggregated_mean_hourly(data, n_days, 'mean', 'loss')

    hourly_gain['timestamp'] = hourly_gain.index
    hourly_loss['timestamp'] = hourly_loss.index
    hourly_data = pd.DataFrame()
    hourly_data['timestamp'] = data['timestamp']
    hourly_data['close'] = data['close']
    hourly_data['avg_gain'] = hourly_gain # maybe merge here based on timestamp
    hourly_data['avg_loss'] = hourly_loss

    hourly_data['avg_loss']  = hourly_data['avg_loss'].replace(0, 1e-10)
    rs = hourly_data['avg_gain'] / hourly_data['avg_loss']
    hourly_data['rsi'] = 100 - (100 / (1 + rs))

    # filter out first x days to account for delay
    hourly_data = hourly_data[hourly_data['timestamp'].dt.date >= start_date.date()]
    # hourly_data = hourly_data[hourly_data['timestamp'] > hourly_data['timestamp'][0]+timedelta(int(n_days.split('D')[0]))]

    # rsi_upward = (hourly_data['rsi'] > low_thresh) & (hourly_data['rsi'].shift(-1) <= low_thresh)
    # rsi_downward = (hourly_data['rsi'] < high_thresh) & (hourly_data['rsi'].shift(-1) >= high_thresh)
    # hourly_data['signal'] = np.where(rsi_upward, 1, np.where(rsi_downward, 0, float("nan")))
    if hourly_data.empty:
        return hourly_data
    hourly_data.loc[hourly_data['rsi'] < low_thresh, 'signal'] = 1
    hourly_data.loc[hourly_data['rsi'] > high_thresh, 'signal'] = 0
    hourly_data['low_thresh'] = low_thresh
    hourly_data['high_thresh'] = high_thresh

    hourly_data['position'] = hourly_data[hourly_data['signal'].notna()]['signal'].diff() #hourly_data['signal'].diff()
    hourly_data['buy'] = np.where(hourly_data['position'] == 1, hourly_data['close'], np.NAN)
    hourly_data['sell'] = np.where(hourly_data['position'] == -1, hourly_data['close'], np.NAN)

    if dir_name is not None:
        plot_original_data_trade_signals_subplots(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'], hourly_data['rsi'], hourly_data['low_thresh'], hourly_data['high_thresh']],
                                        'hourly_rsi',
                                        'Hourly RSI',
                                        ['Original data', 'Hourly RSI', 'low threshold', 'high threshold'],
                                        dir_name,
                                        file_type)
    return hourly_data

def calculate_rsi(data, col, period):
    price_differences = data[col].diff()
    gain = price_differences.where(price_differences > 0, 0)
    # loss = -price_differences.where(price_differences < 0, 0)
    loss = abs(price_differences.where(price_differences < 0, 0))

    # gain = np.where(price_differences > 0, price_differences, 0)
    # loss = np.where(price_differences < 0, -price_differences, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    # data[data['timestamp'].dt.date == pd.to_datetime('2022-01-05')]['close'].count()
    # avg_gain = gain.ewm(span=391*5, adjust=False).mean()
    # avg_loss = loss.ewm(span=391*5, adjust=False).mean()
    avg_loss = avg_loss.replace(0, 1e-10)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# https://www.youtube.com/watch?v=PUk5E8G1r44
def get_buy_and_sell_signals(data, col1, col2, col3='close'):
    data['signal'] = np.where(data[col1] > data[col2], 1, 0)
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
