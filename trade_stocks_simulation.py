from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetExchange, PositionSide, OrderClass, \
OrderType, OrderStatus, AssetStatus, AccountStatus, ActivityType, TradeActivityType, NonTradeActivityStatus, \
CorporateActionType, CorporateActionSubType, CorporateActionDateType, DTBPCheck, PDTCheck, TradeConfirmationEmail
from alpaca.trading.client import TradingClient
from alpaca.data.requests import StockLatestQuoteRequest, StockQuotesRequest
from alpaca.data.historical import StockHistoricalDataClient
import yaml
import time
from datetime import datetime, timezone, date, timedelta
import csv
import time
import pytz
from determine_trade_times import get_buy_and_sell_signals, get_baseline_signals, get_sma_crossover_signal, get_hourly_sma_crossover_signal, \
get_slow_stochastic_oscillator, get_hourly_slow_stochastic_oscillator, get_mean_reversion_signal, get_hourly_mean_reversion_signal, \
get_rsi_signal, get_hourly_rsi_signal
from live_and_sim_trading_helper_functions import UUIDEncoder, write_trade_data_to_file, write_account_info_to_file, get_stock_info, get_stock_quotes, write_trade_info_to_file, simulate_execute_trades
import pandas as pd
from generate_training_data_ml import generate_and_save_training_data, format_training_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import svm
import numpy as np
from ast import literal_eval

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
data_path = params.get("data_path")
account_data_path = params.get("account_data_path")
AV_API_KEY = params.get("alpha_vantage").get("api_key")
training_data_dir = params.get("training_data_directory")
live_sim_training_data_dir = params.get("live_sim_training_data_directory")
raw_data_dir = params.get("raw_data_directory")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

raw_data_t3 = f'{raw_data_dir}/raw_data_t3.csv'
optimal_param_file_t0 = f'{raw_data_dir}/params_t0.csv'

data_t3 = pd.read_csv(raw_data_t3)

optimal_params_df = pd.read_csv(optimal_param_file_t0)
optimal_params = optimal_params_df.groupby(['strategy', 'params']).count().sort_values(by='score', ascending=True).groupby(level=0).tail(1).reset_index() # https://stackoverflow.com/questions/51053911/get-max-of-count-function-on-pandas-groupby-objects

trade_info_file_name = f'{live_sim_training_data_dir}/trade_info_live_sim_final.csv'

strategy_parameters = {

    'sma': {
        'short_time_period': optimal_params[optimal_params['strategy'] == 'sma']['params'].apply(literal_eval).reset_index().loc[0]['params'][0],
        'long_time_period': optimal_params[optimal_params['strategy'] == 'sma']['params'].apply(literal_eval).reset_index().loc[0]['params'][1]
    },
    'sma_hourly': {
        'short_time_period': optimal_params[optimal_params['strategy'] == 'sma_hourly']['params'].apply(literal_eval).reset_index().loc[0]['params'][0],
        'long_time_period': optimal_params[optimal_params['strategy'] == 'sma_hourly']['params'].apply(literal_eval).reset_index().loc[0]['params'][1]
    },
    'stoch': {
        'short_time_period': optimal_params[optimal_params['strategy'] == 'stoch']['params'].apply(literal_eval).reset_index().loc[0]['params'][0],
        'long_time_period': optimal_params[optimal_params['strategy'] == 'stoch']['params'].apply(literal_eval).reset_index().loc[0]['params'][1]
    },
    'stoch_hourly': {
        'short_time_period': optimal_params[optimal_params['strategy'] == 'stoch_hourly']['params'].apply(literal_eval).reset_index().loc[0]['params'][0],
        'long_time_period': optimal_params[optimal_params['strategy'] == 'stoch_hourly']['params'].apply(literal_eval).reset_index().loc[0]['params'][1]
    },
    'mean_rever': {
        'time_period': optimal_params[optimal_params['strategy'] == 'mean_rever']['params'].apply(literal_eval).reset_index().loc[0]['params'][0]
    },
    'mean_rever_hourly': {
        'time_period': optimal_params[optimal_params['strategy'] == 'mean_rever_hourly']['params'].apply(literal_eval).reset_index().loc[0]['params'][0]
    },
    'rsi': {
        'time_period': optimal_params[optimal_params['strategy'] == 'rsi']['params'].apply(literal_eval).reset_index().loc[0]['params'][0]
    },
    'rsi_hourly': {
        'time_period': optimal_params[optimal_params['strategy'] == 'rsi_hourly']['params'].apply(literal_eval).reset_index().loc[0]['params'][0]
    }
}

# decide how to choose from the predicted values which strategy to use
# for live simulation can actually get the results and see if they are correct but in reality we wont be able to do get an accuracy score for live data

start_of_trading_day = datetime(2023, 1, 1, 9, 30, 0).time() # 9:30am EST
end_of_trading_day = datetime(2023, 1, 1, 16, 0, 0).time() # 4:00pm EST
est = pytz.timezone('US/Eastern')

t3_start_date = datetime(2023, 1, 1)
t3_end_date = datetime(2023, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
stock_symbols = ['TSLA', 'TM', 'SBUX', 'AMZN', 'MSFT', 'ORCL', 'AMAT', 'TSM', 'MRNA', 'JNJ', 'WMT', 'COST', 'UNH', 'CVS', 'JPM', 'PYPL', 'PEP', 'KO']
for i in range(0, len(stock_symbols)):
    stock_symbol = stock_symbols[i]
    # strategy = y_pred_random_forest[i][9]
    # if strategy == 'baseline':
    #     continue
    # data_t3
    df = get_stock_info(stock_symbol, t3_start_date, t3_end_date, data_client)
    # use previously saved data instead - in real time would do this

    # Make sure dataframe only has data for timestamps between 9:30am and 4pm EST
    df = df[(df['timestamp'].dt.tz_convert(est).dt.time >= start_of_trading_day) & (df['timestamp'].dt.tz_convert(est).dt.time <= end_of_trading_day)]
    df.index = df['timestamp'].dt.tz_convert(est)

    # Convert to EST timezone
    df.index = df.index.tz_convert(est)
    df['timestamp'] = df['timestamp'].dt.tz_convert(est)

    # Baseline- purely time based
    baseline_buy_signal, baseline_sell_signal = get_baseline_signals(df)
    # baseline_signal = pd.DataFrame(columns=['timestamp', 'close', 'signal', 'buy', 'sell'])
    # baseline_sell_signal.rename(columns={'close':'close_sell'})
    # baseline_sell_signal.rename(columns={'close':'close_sell'})
    baseline_buy_signal['signal'] = 1
    baseline_sell_signal['signal'] = 0
    baseline_buy_signal['buy'] = baseline_buy_signal['close']
    baseline_sell_signal['sell'] = baseline_sell_signal['close']
    baseline_signal = pd.concat([baseline_buy_signal[['close', 'signal', 'buy']], baseline_sell_signal[['close', 'signal', 'sell']]]).sort_values(by='timestamp')
    baseline_signal['timestamp'] = baseline_signal.index

    # SMA Crossover
    crossover_signal = get_sma_crossover_signal(df, t3_start_date, strategy_parameters['sma']['short_time_period'], strategy_parameters['sma']['long_time_period'])
    hourly_mean_crossover_signal = get_hourly_sma_crossover_signal(df, t3_start_date,strategy_parameters['sma_hourly']['short_time_period'], strategy_parameters['sma_hourly']['long_time_period'])

    # Slow Stochastic Oscillator
    slow_stochastic_oscillator = get_slow_stochastic_oscillator(df, t3_start_date, f'{strategy_parameters["stoch"]["long_time_period"]}D', f'{strategy_parameters["stoch"]["short_time_period"]}D', 20, 80)
    slow_stochastic_oscillator_hourly = get_hourly_slow_stochastic_oscillator(df, t3_start_date, f'{strategy_parameters["stoch_hourly"]["long_time_period"]}D', f'{strategy_parameters["stoch_hourly"]["short_time_period"]}D', 20, 80)

    # Mean Reversion Strategy
    mean_reversion_signal = get_mean_reversion_signal(df, t3_start_date, f'{strategy_parameters["mean_rever"]["time_period"]}D', [-1.5, 1.5])
    mean_reversion_signal_hourly = get_hourly_mean_reversion_signal(df, t3_start_date, f'{strategy_parameters["mean_rever_hourly"]["time_period"]}D', [-1.5, 1.5])

    # RSI
    rsi_signal = get_rsi_signal(df, t3_start_date, f'{strategy_parameters["rsi"]["time_period"]}D', 30, 70)
    rsi_signal_hourly = get_hourly_rsi_signal(df, t3_start_date, f'{strategy_parameters["rsi_hourly"]["time_period"]}D', 30, 70)

    simulate_execute_trades(stock_symbol, df, baseline_signal, 'baseline', 1000, 0, trade_info_file_name)
    simulate_execute_trades(stock_symbol, df, crossover_signal, 'sma', 1000, strategy_parameters['sma']['long_time_period'], trade_info_file_name)
    simulate_execute_trades(stock_symbol, df, hourly_mean_crossover_signal, 'sma_hourly', 1000, strategy_parameters['sma_hourly']['long_time_period'], trade_info_file_name)
    simulate_execute_trades(stock_symbol, df, slow_stochastic_oscillator, 'stoch', 1000, strategy_parameters['stoch']['long_time_period'], trade_info_file_name)
    simulate_execute_trades(stock_symbol, df, slow_stochastic_oscillator_hourly, 'stoch_hourly', 1000, strategy_parameters['stoch_hourly']['long_time_period'], trade_info_file_name)
    simulate_execute_trades(stock_symbol, df, mean_reversion_signal, 'mean_rever', 1000, strategy_parameters['mean_rever']['time_period'], trade_info_file_name)
    simulate_execute_trades(stock_symbol, df, mean_reversion_signal_hourly, 'mean_rever_hourly', 1000, strategy_parameters['mean_rever_hourly']['time_period'], trade_info_file_name)
    simulate_execute_trades(stock_symbol, df, rsi_signal, 'rsi', 1000, strategy_parameters['rsi']['time_period'], trade_info_file_name)
    simulate_execute_trades(stock_symbol, df, rsi_signal_hourly, 'rsi_hourly', 1000, strategy_parameters['rsi_hourly']['time_period'], trade_info_file_name)
