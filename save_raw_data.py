from datetime import datetime, timedelta
import pytz
from pathlib import Path
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
import pdb
from stock_split_info import adjust_for_stock_split
from itertools import product
import csv
import pandas as pd
import yaml
from alpaca.data.historical import StockHistoricalDataClient
from generate_training_data_ml import save_company_data, get_stock_symbols, get_income_statement_data, get_and_save_raw_data
import time

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
AV_API_KEY = params.get("alpha_vantage").get("api_key")
year = params.get("year")
stock_list_dir = params.get("stock_list_directory")
raw_data_dir = params.get("raw_data_directory")
training_data_dir = params.get("training_data_directory")

file_name_t0 = f'{raw_data_dir}/raw_data_t0.csv'
file_name_t1 = f'{raw_data_dir}/raw_data_t1.csv'
file_name_t2 = f'{raw_data_dir}/raw_data_t2.csv'
file_name_t3 = f'{raw_data_dir}/raw_data_t3.csv'
company_data_file = f'{training_data_dir}/company_data.csv'
annual_report_file = f'{training_data_dir}/annual_reports.csv'
quarterly_report_file = f'{training_data_dir}/quarterly_reports.csv'

est = pytz.timezone('US/Eastern')
t0_start_date = datetime(year-1, 7, 1)
t0_end_date = datetime(year-1, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t1_start_date = datetime(year, 1, 1)
t1_end_date = datetime(year, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t2_start_date = datetime(year, 7, 1)
t2_end_date = datetime(year, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t3_start_date = datetime(year+1, 1, 1)
t3_end_date = datetime(year+1, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Get symbols for training data
print('Saving raw data')
nasdaq_symbols = get_stock_symbols(400, f'{stock_list_dir}/custom-nasdaq-stocks-stocks-all.csv')
nyse_symbols = get_stock_symbols(400, f'{stock_list_dir}/custom-nyse-stocks-stocks-all.csv')
save_company_data(pd.concat([nasdaq_symbols, nyse_symbols]), company_data_file)

stock_symbols = pd.read_csv(company_data_file)["Symbol"].tolist()

# Collect income data on symbols
print('Saving income statement data for 800 stocks')
get_income_statement_data(stock_symbols, annual_report_file, quarterly_report_file, AV_API_KEY)

print(f'Getting data for t0: {t0_start_date} to {t0_end_date}')
start = time.time()
get_and_save_raw_data(stock_symbols,
                      data_client,
                      t0_start_date-timedelta(days=50),
                      t0_end_date,
                      file_name_t0
                     )
end = time.time()
print(f'Time to collect data for t0: {(end - start)/60} minutes')

print(f'Getting data for t1: {t1_start_date} to {t1_end_date}')
start = time.time()
get_and_save_raw_data(stock_symbols,
                      data_client,
                      t1_start_date-timedelta(days=50),
                      t1_end_date,
                      file_name_t1
                     )
end = time.time()
print(f'Time to collect data for t1: {(end - start)/60} minutes')

print(f'Getting data for t2: {t2_start_date} to {t2_end_date}')
start = time.time()
get_and_save_raw_data(stock_symbols,
                      data_client,
                      t2_start_date-timedelta(days=50),
                      t2_end_date,
                      file_name_t2
                     )
end = time.time()
print(f'Time to collect data for t2: {(end - start)/60} minutes')

print(f'Getting data for t3: {t3_start_date} to {t3_end_date}')
start = time.time()
get_and_save_raw_data(stock_symbols,
                      data_client,
                      t3_start_date-timedelta(days=50),
                      t3_end_date,
                      file_name_t3
                     )
end = time.time()
print(f'Time to collect data for t3: {(end - start)/60} minutes')
