import numpy as np
import pandas as pd
from plot_original_data import plot_original_data_trade_signals, plot_original_data_trade_signals_subplots, plot_original_data_year_with_trade_markers

def sma(data, col, window):
    return data[col].rolling(window=window).mean()

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
                                        [data['close'], stoch_osc['%K'], stoch_osc['low_thresh'], stoch_osc['high_thresh'], stoch_osc['%D']],
                                        'slow_stoch',
                                        'Slow Stochastic Oscillator',
                                        ['Original data', '%K line', 'low threshold', 'high threshold', '%D line'],
                                        dir_name,
                                        file_type
                                        )

    return stoch_osc

def get_hourly_slow_stochastic_oscillator(data, start_date, k, d, low_thresh, high_thresh, dir_name=None, file_type=None):
    # Group data by day and hour
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

    # Calculate the %K line
    hourly_osc['%K'] = ((hourly_osc['close'] - hourly_osc['lowest_low']) / (hourly_osc['highest_high'] - hourly_osc['lowest_low'] + 1e-10)) * 100
    # Calculate the %D line
    hourly_osc['%D'] = hourly_osc['%K'].rolling(d).mean()

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
                                        [data['close'], hourly_osc['%K'], hourly_osc['low_thresh'], hourly_osc['high_thresh'], hourly_osc['%D']],
                                        'hourly_slow_stoch',
                                        'Hourly Slow Stoc. Osc.',
                                        ['Original data', 'Hourly %K line', 'low threshold', 'high threshold', 'Hourly %D line'],
                                        dir_name,
                                        file_type)

    return hourly_osc

def get_mean_reversion_signal(data, start_date, n_days, threshold, dir_name=None, file_type=None):
    new_data = data.copy(deep=True)
    new_data['close_mean'] = new_data['close'].rolling(n_days).mean()
    new_data['deviation'] = new_data['close'] - new_data['close_mean']
    new_data['std_dev'] = (new_data['deviation'].pow(2)/new_data['close'].rolling(n_days).count()).pow(1/2)
    new_data['std_dev']  = new_data['std_dev'].replace(0, 1e-10)
    new_data['z_score'] = new_data['deviation']/new_data['std_dev']

    # filter out first x days to account for delay
    new_data = new_data[new_data['timestamp'].dt.date >= start_date.date()]

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

    if hourly_data.empty:
        return hourly_data

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

def get_rsi_signal(data, start_date, n_days, low_thresh, high_thresh, col='close', dir_name=None, file_type=None):
    new_data = data.copy(deep=True)
    new_data['rsi'] = calculate_rsi(data, col, n_days)

    if new_data.empty:
        return new_data
    new_data.loc[new_data['rsi'] < low_thresh, 'signal'] = 1
    new_data.loc[new_data['rsi'] > high_thresh, 'signal'] = 0
    new_data['low_thresh'] = low_thresh
    new_data['high_thresh'] = high_thresh

    # filter out first x days to account for delay
    new_data = new_data[new_data['timestamp'].dt.date >= start_date.date()]

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
    loss = abs(price_differences.where(price_differences < 0, 0))

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
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
def get_baseline_signals(data, dir_name=None, file_type=None):
    baseline_buy_signal = data.groupby(data.index.date).apply(lambda x: x.iloc[[0]]) # first datapoint from each day
    baseline_sell_signal = data.groupby(data.index.date).apply(lambda x: x.iloc[[-1]]) # last datapoint from each day
    baseline_buy_signal.index = baseline_buy_signal.index.droplevel(0)
    baseline_sell_signal.index = baseline_sell_signal.index.droplevel(0)

    if dir_name is not None:
        plot_original_data_year_with_trade_markers(data['symbol'][0],
                                        data['timestamp'][0].year,
                                        data['timestamp'],
                                        [data['close'], baseline_buy_signal['close'], baseline_sell_signal['close']],
                                        ['Original data', 'buy', 'sell'],
                                        dir_name,
                                        file_type)

    return baseline_buy_signal, baseline_sell_signal
