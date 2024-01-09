from scipy.optimize import minimize
from itertools import product
from datetime import datetime
import pandas as pd
import yaml
import pdb
from determine_trade_times import get_sma_crossover_signal, get_hourly_sma_crossover_signal, get_slow_stochastic_oscillator, get_hourly_slow_stochastic_oscillator,\
get_mean_reversion_signal, get_hourly_mean_reversion_signal, get_rsi_signal, get_hourly_rsi_signal
from analyze_trades import determine_profits, format_trade_signals
import pytz
import csv
from pathlib import Path


def custom_profit_scorer(raw_data, key, params):
    strategy_functions = {
        'sma': get_sma_crossover_signal(raw_data, params[0], params[-1]),
        'sma_hourly': get_hourly_sma_crossover_signal(df, params[0], params[-1]),
        'stoch': get_slow_stochastic_oscillator(df, f'{params[-1]}D', f'{params[0]}D', 20, 80),
        'stoch_hourly': get_hourly_slow_stochastic_oscillator(df, f'{params[-1]}D', f'{params[0]}D', 20, 80),
        'mean_rever': get_mean_reversion_signal(df, f'{params[0]}D', [-1.5, 1.5]),
        'mean_rever_hourly': get_hourly_mean_reversion_signal(df, f'{params[0]}D', [-1.5, 1.5]),
        'rsi': get_rsi_signal(df, f'{params[0]}D', 30, 70),
        'rsi_hourly': get_hourly_rsi_signal(df, f'{params[0]}D', 30, 70)
    }

    trade_strategy_signals = strategy_functions[key]
    trade_signals = format_trade_signals(trade_strategy_signals)
    total_profits_signal, returns_signal = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])

    return returns_signal


def optimize_parameters(raw_data, key, parameter_grid):
    # Generate all combinations of parameters
    parameter_combinations = list(product(*parameter_grid.values()))
    best_params = None
    best_score = float('-inf')

    # Iterate over parameter combinations
    for params in parameter_combinations:
        # Apply the trading strategies with specific parameter values
        # Calculate the performance score (total profits) on the training data
        score = custom_profit_scorer(raw_data, key, params)

        # Update best parameters if the current combination has a higher score
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
year = params.get("year")
raw_data_dir = params.get("raw_data_directory")
training_data_dir = params.get("training_data_directory")

file_name_t0 = f'{raw_data_dir}/raw_data_t0.csv'
file_name_t1 = f'{raw_data_dir}/raw_data_t1.csv'
file_name_t2 = f'{raw_data_dir}/raw_data_t2.csv'
file_name_t3 = f'{raw_data_dir}/raw_data_t3.csv'

optimal_param_file_t0 = f'{raw_data_dir}/params_t0.csv'
optimal_param_file_t1 = f'{raw_data_dir}/params_t1.csv'
optimal_param_file_t2 = f'{raw_data_dir}/params_t2.csv'
optimal_param_file_t3 = f'{raw_data_dir}/params_t3.csv'

start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST
est = pytz.timezone('US/Eastern')

data = pd.read_csv(file_name_t0)
stock_symbols = data['symbol'].unique().tolist()

# Define the parameter grid for grid search
parameter_grids = {

    'sma': {
        'short_time_period': [5, 10, 20],
        'long_time_period': [30, 40, 50]
    },
    'sma_hourly': {
        'short_time_period': [5, 10, 20],
        'long_time_period': [30, 40, 50]
    },
    'stoch': {
        'short_time_period': [1, 2, 3],
        'long_time_period': [5, 9, 14]
        # 'low_threshold': [15, 20, 25],
        # 'high_threshold': [75, 80, 85]
    },
    'stoch_hourly': {
        'short_time_period': [1, 2, 3],
        'long_time_period': [5, 9, 14]
        # 'low_threshold': [15, 20, 25],
        # 'high_threshold': [75, 80, 85]
    },
    'mean_rever': {
        'time_period': [5, 10, 20]
        # 'low_threshold': [-2, -1.5, -1],
        # 'high_threshold': [1, 1.5, 2]
    },
    'mean_rever_hourly': {
        'time_period': [5, 10, 20]
        # 'low_threshold': [-2, -1.5, -1],
        # 'high_threshold': [1, 1.5, 2]
    },
    'rsi': {
        'time_period': [3, 10, 20]
        # 'low_threshold': [25, 30, 35],
        # 'high_threshold': [65, 70, 75]
    },
    'rsi_hourly': {
        'time_period': [3, 10, 20]
        # 'low_threshold': [25, 30, 35],
        # 'high_threshold': [65, 70, 75]
    }
}


param_columns=['symbol', 'strategy', 'params', 'score']
if not Path(optimal_param_file_t0).exists():
    with open(optimal_param_file_t0, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(param_columns)

for symbol in stock_symbols:
    if not Path(optimal_param_file_t0).exists():
        existing_symbols = []
    else:
        existing_symbols = pd.read_csv(optimal_param_file_t0)["symbol"].tolist()

    if symbol in existing_symbols:
        print(f"Data already exists for {symbol}")
        continue

    print(symbol)
    params_df_symbol = pd.DataFrame(columns=param_columns)
    df = data[data['symbol'] == symbol]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'].dt.tz_convert(est).dt.time >= start_of_trading_day) & (df['timestamp'].dt.tz_convert(est).dt.time <= end_of_trading_day)]
    df.index = df['timestamp']

    for key in parameter_grids:
        optimized_params, optimized_score = optimize_parameters(df, key, parameter_grids[key])
        params_df_symbol.loc[len(params_df_symbol)] = [symbol, key, optimized_params, optimized_score]
    params_df_symbol.to_csv(optimal_param_file_t0, mode='a', header=False, index=False)

# params_df['params'].value_counts()
# params_df.groupby(['strategy', 'params']).count()
