from alpaca.data.historical import StockHistoricalDataClient
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import yfinance as yf
from datetime import datetime, timezone, date, timedelta
from generate_training_data_ml import get_stock_symbols, generate_and_save_training_data, format_training_data, get_company_data, save_company_data, get_income_statement_data, generate_training_data
import requests
import time
import csv
import pytz
from pathlib import Path
from statistics import mean
import dataframe_image as dfi
from ast import literal_eval
from ml_helper_functions import run_ml_models, normalize_data

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
AV_API_KEY = params.get("alpha_vantage").get("api_key")
year = params.get("year")
training_data_dir = params.get("training_data_directory")
tables_dir_name = params.get("ml_tables_directory")
stock_list_dir = params.get("stock_list_directory")
raw_data_dir = params.get("raw_data_directory")

raw_data_t0 = f'{raw_data_dir}/raw_data_t0.csv'
raw_data_t1 = f'{raw_data_dir}/raw_data_t1.csv'
raw_data_t2 = f'{raw_data_dir}/raw_data_t2.csv'
raw_data_t3 = f'{raw_data_dir}/raw_data_t3.csv'
optimal_param_file_t0 = f'{raw_data_dir}/params_t0.csv'
optimal_param_file_t1 = f'{raw_data_dir}/params_t1.csv'
file_name_t0 = f'{training_data_dir}/stock_strategy_data_t0.csv'
file_name_t1 = f'{training_data_dir}/stock_strategy_data_t1.csv'
file_name_t2 = f'{training_data_dir}/stock_strategy_data_t2.csv'
file_name_t3 = f'{training_data_dir}/stock_strategy_data_t3.csv'
file_name_hourly_t0 = f'{training_data_dir}/stock_strategy_data_hourly_t0.csv'
file_name_hourly_t1 = f'{training_data_dir}/stock_strategy_data_hourly_t1.csv'
file_name_hourly_t2 = f'{training_data_dir}/stock_strategy_data_hourly_t2.csv'
file_name_hourly_t3 = f'{training_data_dir}/stock_strategy_data_hourly_t3.csv'

company_data_file = f'{training_data_dir}/company_data.csv'
annual_report_file = f'{training_data_dir}/annual_reports.csv'
quarterly_report_file = f'{training_data_dir}/quarterly_reports.csv'

col_names=(['symbol','baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof',
            'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof',
            'baseline_profits','sma_profits','sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits',
            'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits',
            'baseline_returns','sma_returns','sma_hourly_returns', 'stoch_returns', 'stoch_hourly_returns',
            'mean_rever_returns', 'mean_rever_hourly_returns', 'rsi_returns', 'rsi_hourly_returns',
            'baseline_total_buys','sma_total_buys','sma_hourly_total_buys', 'stoch_total_buys', 'stoch_hourly_total_buys',
            'mean_rever_total_buys', 'mean_rever_hourly_total_buys', 'rsi_total_buys', 'rsi_hourly_total_buys',
            'baseline_total_sells','sma_total_sells','sma_hourly_total_sells', 'stoch_total_sells', 'stoch_hourly_total_sells',
            'mean_rever_total_sells', 'mean_rever_hourly_total_sells', 'rsi_total_sells', 'rsi_hourly_total_sells',
            'mean_price','std_dev', 'low', 'high', 'best_strategy'
])

col_names_hourly=(['symbol','hour','baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof',
            'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof',
            'baseline_is_prof_buy','sma_is_prof_buy','sma_hourly_is_prof_buy', 'stoch_is_prof_buy', 'stoch_hourly_is_prof_buy',
            'mean_rever_is_prof_buy', 'mean_rever_hourly_is_prof_buy', 'rsi_is_prof_buy', 'rsi_hourly_is_prof_buy',
            'baseline_profits','sma_profits','sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits',
            'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits',
            'baseline_profits_buy','sma_profits_buy','sma_hourly_profits_buy', 'stoch_profits_buy', 'stoch_hourly_profits_buy',
            'mean_rever_profits_buy', 'mean_rever_hourly_profits_buy', 'rsi_profits_buy', 'rsi_hourly_profits_buy',
            'baseline_returns','sma_returns','sma_hourly_returns', 'stoch_returns', 'stoch_hourly_returns',
            'mean_rever_returns', 'mean_rever_hourly_returns', 'rsi_returns', 'rsi_hourly_returns',
            'baseline_returns_buy','sma_returns_buy','sma_hourly_returns_buy', 'stoch_returns_buy', 'stoch_hourly_returns_buy',
            'mean_rever_returns_buy', 'mean_rever_hourly_returns_buy', 'rsi_returns_buy', 'rsi_hourly_returns_buy',
            'baseline_total_sells','sma_total_sells','sma_hourly_total_sells', 'stoch_total_sells', 'stoch_hourly_total_sells',
            'mean_rever_total_sells', 'mean_rever_hourly_total_sells', 'rsi_total_sells', 'rsi_hourly_total_sells',
            'baseline_total_buys','sma_total_buys','sma_hourly_total_buys', 'stoch_total_buys', 'stoch_hourly_total_buys',
            'mean_rever_total_buys', 'mean_rever_hourly_total_buys', 'rsi_total_buys', 'rsi_hourly_total_buys',
            'mean_price', 'std_dev', 'low', 'high', 'best_strategy_sell', 'best_strategy_buy'
])

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
est = pytz.timezone('US/Eastern')

start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST

# Get best parameters for each strategy based on t0
optimal_params_df = pd.read_csv(optimal_param_file_t0)
optimal_params = optimal_params_df.groupby(['strategy', 'params']).count().sort_values(by='score', ascending=True).groupby(level=0).tail(1).reset_index() # https://stackoverflow.com/questions/51053911/get-max-of-count-function-on-pandas-groupby-objects

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

data_files = {
    't0': [raw_data_t0, file_name_t0, file_name_hourly_t0],
    't1': [raw_data_t1, file_name_t1, file_name_hourly_t1],
    't2': [raw_data_t2, file_name_t2, file_name_hourly_t2],
    't3': [raw_data_t3, file_name_t3, file_name_hourly_t3]
}

t0_start_date = datetime(year-1, 7, 1)
t0_end_date = datetime(year-1, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t1_start_date = datetime(year, 1, 1)
t1_end_date = datetime(year, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t2_start_date = datetime(year, 7, 1)
t2_end_date = datetime(year, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t3_start_date = datetime(year+1, 1, 1)
t3_end_date = datetime(year+1, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)

time_period_dates = {
    't0': {'start': t0_start_date, 'end': t0_end_date},
    't1': {'start': t1_start_date, 'end': t1_end_date},
    't2': {'start': t2_start_date, 'end': t2_end_date},
    't3': {'start': t3_start_date, 'end': t3_end_date}
}

start = time.time()
for time_period in data_files:
    print(f'Getting data for {time_period}')
    data = pd.read_csv(data_files[time_period][0])
    stock_symbols = data['symbol'].unique().tolist()
    for symbol in stock_symbols:
        df = data[data['symbol'] == symbol]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'].dt.tz_convert(est).dt.time >= start_of_trading_day) & (df['timestamp'].dt.tz_convert(est).dt.time <= end_of_trading_day)]
        df.index = df['timestamp']
        generate_training_data(df, symbol, time_period_dates[time_period]['start'], strategy_parameters, data_files[time_period][1], data_files[time_period][2], col_names, col_names_hourly)
end = time.time()
print(f'Time to generate training data: {(end-start)/60} Minutes')

labels = ['baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof',
            'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof', 'best_strategy']
labels_hourly = ['baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof',
            'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof',
            'baseline_is_prof_buy','sma_is_prof_buy','sma_hourly_is_prof_buy', 'stoch_is_prof_buy', 'stoch_hourly_is_prof_buy',
            'mean_rever_is_prof_buy', 'mean_rever_hourly_is_prof_buy', 'rsi_is_prof_buy', 'rsi_hourly_is_prof_buy',
            'best_strategy_sell', 'best_strategy_buy']

# Company data and reports
company_data = pd.read_csv(company_data_file)
annual_reports = pd.read_csv(annual_report_file)
quarterly_reports = pd.read_csv(quarterly_report_file)

# profitable strategies per stock
profitable_strategies_t0 = pd.read_csv(file_name_t0)
profitable_strategies_t1 = pd.read_csv(file_name_t1)
profitable_strategies_t2 = pd.read_csv(file_name_t2)
profitable_strategies_t3 = pd.read_csv(file_name_t3)

extra_t0 = list(set(profitable_strategies_t0['symbol'].tolist()) - set(profitable_strategies_t1['symbol'].tolist())) + list(set(profitable_strategies_t0['symbol'].tolist()) - set(profitable_strategies_t2['symbol'].tolist()))
extra_t1 = list(set(profitable_strategies_t1['symbol'].tolist()) - set(profitable_strategies_t0['symbol'].tolist())) + list(set(profitable_strategies_t1['symbol'].tolist()) - set(profitable_strategies_t2['symbol'].tolist()))
extra_t2 = list(set(profitable_strategies_t2['symbol'].tolist()) - set(profitable_strategies_t0['symbol'].tolist())) + list(set(profitable_strategies_t2['symbol'].tolist()) - set(profitable_strategies_t1['symbol'].tolist()))

profitable_strategies_t0 = profitable_strategies_t0[~profitable_strategies_t0['symbol'].isin(extra_t0)]
profitable_strategies_t1 = profitable_strategies_t1[~profitable_strategies_t1['symbol'].isin(extra_t1)]
profitable_strategies_t2 = profitable_strategies_t2[~profitable_strategies_t2['symbol'].isin(extra_t2)]

# profitable strategies per hour per stock
profitable_strategies_hourly_t0 = pd.read_csv(file_name_hourly_t0)
profitable_strategies_hourly_t1 = pd.read_csv(file_name_hourly_t1)
profitable_strategies_hourly_t2 = pd.read_csv(file_name_hourly_t2)
profitable_strategies_hourly_t3 = pd.read_csv(file_name_hourly_t3)

extra_t0_hourly = list(set(profitable_strategies_hourly_t0['symbol'].tolist()) - set(profitable_strategies_hourly_t1['symbol'].tolist())) + list(set(profitable_strategies_hourly_t0['symbol'].tolist()) - set(profitable_strategies_hourly_t2['symbol'].tolist()))
extra_t1_hourly = list(set(profitable_strategies_hourly_t1['symbol'].tolist()) - set(profitable_strategies_hourly_t0['symbol'].tolist())) + list(set(profitable_strategies_hourly_t1['symbol'].tolist()) - set(profitable_strategies_hourly_t2['symbol'].tolist()))
extra_t2_hourly = list(set(profitable_strategies_hourly_t2['symbol'].tolist()) - set(profitable_strategies_hourly_t0['symbol'].tolist())) + list(set(profitable_strategies_hourly_t2['symbol'].tolist()) - set(profitable_strategies_hourly_t1['symbol'].tolist()))

profitable_strategies_hourly_t0 = profitable_strategies_hourly_t0[~profitable_strategies_hourly_t0['symbol'].isin(extra_t0_hourly)]
profitable_strategies_hourly_t1 = profitable_strategies_hourly_t1[~profitable_strategies_hourly_t1['symbol'].isin(extra_t1_hourly)]
profitable_strategies_hourly_t2 = profitable_strategies_hourly_t2[~profitable_strategies_hourly_t2['symbol'].isin(extra_t2_hourly)]

quarterly_reports_df = quarterly_reports[['fiscalDateEnding', 'symbol', 'totalRevenue', 'ebitda', 'ebit']]
quarterly_reports_df[['totalRevenue', 'ebitda', 'ebit']] = quarterly_reports_df[['totalRevenue', 'ebitda', 'ebit']].apply(pd.to_numeric, errors='coerce')

# select columns to use from company_data and quarterly_reports, combine with profitable strategies
training_data_t0 = format_training_data(profitable_strategies_t0, company_data, quarterly_reports_df, t0_start_date, t0_end_date)
training_data_t1 = format_training_data(profitable_strategies_t1, company_data, quarterly_reports_df, t1_start_date, t1_end_date)
training_data_t2 = format_training_data(profitable_strategies_t2, company_data, quarterly_reports_df, t2_start_date, t2_end_date)
training_data_t3 = format_training_data(profitable_strategies_t3, company_data, quarterly_reports_df, t3_start_date, t3_end_date)

training_data_hourly_t0 = format_training_data(profitable_strategies_hourly_t0, company_data, quarterly_reports_df, t0_start_date, t0_end_date)
training_data_hourly_t1 = format_training_data(profitable_strategies_hourly_t1, company_data, quarterly_reports_df, t1_start_date, t1_end_date)
training_data_hourly_t2 = format_training_data(profitable_strategies_hourly_t2, company_data, quarterly_reports_df, t2_start_date, t2_end_date)
training_data_hourly_t3 = format_training_data(profitable_strategies_hourly_t3, company_data, quarterly_reports_df, t3_start_date, t3_end_date)

training_data_t0.set_index('symbol', inplace=True)
training_data_t1.set_index('symbol', inplace=True)
training_data_t2.set_index('symbol', inplace=True)
training_data_t3.set_index('symbol', inplace=True)

training_data_hourly_t0.set_index('symbol', inplace=True)
training_data_hourly_t1.set_index('symbol', inplace=True)
training_data_hourly_t2.set_index('symbol', inplace=True)
training_data_hourly_t3.set_index('symbol', inplace=True)

training_data = training_data_t1[labels].merge(training_data_t0.drop(labels, axis=1), how='left', right_on='symbol', left_on='symbol')
testing_data = training_data_t2[labels].merge(training_data_t1.drop(labels, axis=1), how='left', right_on='symbol', left_on='symbol')

training_data_hourly = training_data_hourly_t1[['hour']+labels_hourly].merge(training_data_hourly_t0.drop(labels_hourly, axis=1), how='left', right_on=['symbol','hour'], left_on=['symbol','hour'])
testing_data_hourly = training_data_hourly_t2[['hour']+labels_hourly].merge(training_data_hourly_t1.drop(labels_hourly, axis=1), how='left', right_on=['symbol','hour'], left_on=['symbol','hour'])

# Normalize and scale features
normalized_training_data = normalize_data(training_data, labels)
normalized_testing_data = normalize_data(testing_data, labels)
normalized_training_data_hourly = normalize_data(training_data_hourly, labels_hourly)
normalized_testing_data_hourly = normalize_data(testing_data_hourly, labels_hourly)

thresh = 5
non_zero_count = (normalized_training_data != 0).sum()
filtered_columns = non_zero_count[non_zero_count > thresh].index
filtered_normalized_training_data = normalized_training_data[filtered_columns]

non_zero_count = (normalized_testing_data != 0).sum()
filtered_columns = non_zero_count[non_zero_count > thresh].index
filtered_normalized_testing_data = normalized_testing_data[filtered_columns]

thresh = 10
non_zero_count = (normalized_training_data_hourly != 0).sum()
filtered_columns = non_zero_count[non_zero_count > thresh].index
filtered_normalized_training_data_hourly = normalized_training_data_hourly[filtered_columns]

non_zero_count = (normalized_testing_data_hourly != 0).sum()
filtered_columns = non_zero_count[non_zero_count > thresh].index
filtered_normalized_testing_data_hourly = normalized_testing_data_hourly[filtered_columns]

# save original and normalized training data
training_data.to_csv(f'{tables_dir_name}/unscaled_training_data.csv', mode='a', header=True, index=False)
testing_data.to_csv(f'{tables_dir_name}/unscaled_testing_data.csv', mode='a', header=True, index=False)
training_data_hourly.to_csv(f'{tables_dir_name}/unscaled_training_data_hourly.csv', mode='a', header=True, index=False)
testing_data_hourly.to_csv(f'{tables_dir_name}/unscaled_testing_data_hourly.csv', mode='a', header=True, index=False)

normalized_training_data.to_csv(f'{tables_dir_name}/normalized_training_data.csv', mode='a', header=True, index=False)
normalized_testing_data.to_csv(f'{tables_dir_name}/normalized_testing_data.csv', mode='a', header=True, index=False)
normalized_training_data_hourly.to_csv(f'{tables_dir_name}/normalized_training_data_hourly.csv', mode='a', header=True, index=False)
normalized_testing_data_hourly.to_csv(f'{tables_dir_name}/normalized_testing_data_hourly.csv', mode='a', header=True, index=False)

run_ml_models(filtered_normalized_training_data, filtered_normalized_testing_data, labels, tables_dir_name)
run_ml_models(filtered_normalized_training_data_hourly, filtered_normalized_testing_data_hourly, labels_hourly, tables_dir_name, 'hourly')
