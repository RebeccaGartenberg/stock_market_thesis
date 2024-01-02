from alpaca.data.requests import StockLatestQuoteRequest, StockQuotesRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import yaml
import pdb
from datetime import datetime, timezone, date, timedelta
import pytz
import matplotlib.pyplot as plt
import pandas as pd
from determine_trade_times import get_buy_and_sell_signals, get_baseline_signals, get_sma_crossover_signal, get_hourly_sma_crossover_signal, \
get_slow_stochastic_oscillator, get_hourly_slow_stochastic_oscillator, get_mean_reversion_signal, get_hourly_mean_reversion_signal, \
get_rsi_signal, get_hourly_rsi_signal
from aggregate_data import merge_data, offset_data_by_business_days, get_aggregated_mean, get_aggregated_mean_hourly
from plot_stock_data import plot
from analyze_trades import determine_profits, get_total_trades, get_total_trades_per_hour, get_total_profits_per_hour, format_trade_signals
import dataframe_image as dfi
from stock_split_info import adjust_for_stock_split
import random
import csv
import numpy as np
import time
from pathlib import Path
import requests

def save_company_data(company_data, file_name):
    if not Path(file_name).exists():
        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(company_data.columns.tolist())

    company_data.to_csv(file_name, mode='a', header=False, index=False)
    return

def get_income_statement_data(stock_symbols, file_name_1, file_name_2, AV_API_KEY):
    # Get Company Info using Alpha Vantage

    if not Path(file_name_1).exists():
        existing_symbols = []
    else:
        existing_symbols = pd.read_csv(file_name_1)["symbol"].tolist()

    endpoint = f"https://www.alphavantage.co/query"
    function = 'INCOME_STATEMENT'

    for stock_symbol in stock_symbols:
        if stock_symbol in existing_symbols:
            continue

        params = {
            'function': function,
            'symbol': stock_symbol,
            'apikey': AV_API_KEY,
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            income_reports_dict = response.json()
            if income_reports_dict == {}:
                continue
            annual_reports = pd.DataFrame.from_dict(income_reports_dict['annualReports'])
            quarterly_reports = pd.DataFrame.from_dict(income_reports_dict['quarterlyReports'])
        else:
            print(f"Error: {response.status_code} - Unable to retrieve income statement data for {stock_symbol}")

        time.sleep(3) # can only make 5 api calls per minute and 100 per day on free tier

        annual_reports['symbol'] = income_reports_dict['symbol']
        quarterly_reports['symbol'] = income_reports_dict['symbol']

        if not Path(file_name_1).exists():
            with open(file_name_1, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(annual_reports.columns.tolist())

        if not Path(file_name_2).exists():
            with open(file_name_2, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(quarterly_reports.columns.tolist())

        annual_reports.to_csv(file_name_1, mode='a', header=False, index=False)
        quarterly_reports.to_csv(file_name_2, mode='a', header=False, index=False)

    return annual_reports, quarterly_reports

def get_company_data(stock_symbols, file_name, AV_API_KEY):
    # Get Company Info using Alpha Vantage

    if not Path(file_name).exists():
        existing_symbols = []
    else:
        existing_symbols = pd.read_csv(file_name)["Symbol"].tolist()

    endpoint = f"https://www.alphavantage.co/query"
    function = 'OVERVIEW'

    for stock_symbol in stock_symbols:
        if stock_symbol in existing_symbols:
            continue

        params = {
            'function': function,
            'symbol': stock_symbol,
            'apikey': AV_API_KEY,
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            company_data_dict = response.json()
            company_data = pd.DataFrame.from_dict(company_data_dict, orient='index').T
            if 'Symbol' not in company_data:
                continue
            # company_data.set_index('Symbol', inplace=True)
        else:
            print(f"Error: {response.status_code} - Unable to retrieve company data")
            continue

        time.sleep(3) # can only make 5 api calls per minute and 100 per day on free tier

        if not Path(file_name).exists():
            with open(file_name, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(company_data.columns.tolist())

        company_data.to_csv(file_name, mode='a', header=False, index=False)

    return company_data

def format_training_data(profitable_strategies, company_data, quarterly_reports_df, start_date, end_date):
    #  Rating, Top Rating, Founded, IPO Date, 'Market Cap'
    features = ['Symbol', 'Industry', 'Exchange', 'Div. Yield', 'Sector', 'Country', 'State', 'Employees', 'Founded', 'IPO Date', 'IPO Price']
    company_data = company_data[features]

    training_data = profitable_strategies.merge(company_data[features], how='left', left_on='symbol', right_on='Symbol')
    training_data["gives_dividend"] = (training_data["Div. Yield"].notna()).astype(int)
    training_data["ipo_year"] = pd.to_datetime(training_data['IPO Date']).dt.year

    # Columns that need to be encoded: AssetType, Exchange, Currency, Country, Sector, Industry, state
    exchange_cols = pd.get_dummies(training_data["Exchange"], prefix="exchange")
    country_cols = pd.get_dummies(training_data["Country"], prefix="country")
    sector_cols = pd.get_dummies(training_data["Sector"], prefix="sector")
    industry_cols = pd.get_dummies(training_data["Industry"], prefix="industry")
    state_cols = pd.get_dummies(training_data["State"], prefix="state")

    training_data = pd.concat([training_data, exchange_cols, country_cols, sector_cols, industry_cols, state_cols], axis=1)
    training_data = training_data.drop(['State', 'Exchange', 'Country', 'Sector', 'Industry', 'Symbol', 'IPO Date', 'Div. Yield'], axis=1)

    # see about using more columns from quarterly or annual reports and if they should be summed, averaged etc.
    training_data['ebit'] = float('nan')
    training_data['ebitda'] = float('nan')
    training_data['totalRevenue'] = float('nan')
    for symbol in quarterly_reports_df['symbol']:
        filtered_quarterly = quarterly_reports_df[(pd.to_datetime(quarterly_reports_df["fiscalDateEnding"]) > pd.to_datetime(start_date)) & (pd.to_datetime(quarterly_reports_df["fiscalDateEnding"]) <= pd.to_datetime(end_date.date())) & (quarterly_reports_df['symbol'] == symbol)]
        training_data.loc[training_data['symbol'] == symbol, 'ebit'] = filtered_quarterly['ebit'].sum()
        training_data.loc[training_data['symbol'] == symbol, 'ebitda'] = filtered_quarterly['ebitda'].sum()
        training_data.loc[training_data['symbol'] == symbol, 'totalRevenue'] = filtered_quarterly['totalRevenue'].sum()

    return training_data

def format_data(trade_signals, is_profitable, total_profits, total_trades, hourly_df, strategy):
    total_profits_signal, percent_change_signal = determine_profits(trade_signals['buy'].shift(1), trade_signals['sell'])
    is_profitable.append(int(total_profits_signal > 0))
    total_profits.append(total_profits_signal)
    total_trades.append(get_total_trades(trade_signals))
    buy_counts, sell_counts = get_total_trades_per_hour(trade_signals)
    hourly_df[f'{strategy}_total_trades'] = buy_counts + sell_counts
    hourly_profits = get_total_profits_per_hour(trade_signals['buy'].shift(1), trade_signals['sell'])
    hourly_df[f'{strategy}_profits'] = hourly_profits
    hourly_df[f'{strategy}_is_prof'] = (hourly_profits > 0).astype(int)

    return total_profits_signal

def get_stock_symbols(n, file, list=False):
    # symbol_df = pd.read_csv('./stock_lists/stock_symbols.csv', error_bad_lines=False)
    symbol_df = pd.read_csv(file)

    if list:
        symbols = symbol_df["Symbol"].values
        random.shuffle(symbols)
        if n > len(symbols):
            n = len(symbols)
        return symbols[0:n].tolist()

    # min(n,len(pd.read_csv(file).values))
    return symbol_df.loc[0:min(n,len(symbol_df.values))]
    # return symbol_df.sample(frac=1).reset_index(drop=True).iloc[0:n]

def generate_and_save_training_data(stock_symbols, data_client, start_date, end_date, file_name_1, file_name_2, col_names, col_names_hourly, AV_API_KEY):
    start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
    end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST
    est = pytz.timezone('US/Eastern')
    year = start_date.year

    if not Path(file_name_1).exists():
        existing_symbols = []
    else:
        existing_symbols = pd.read_csv(file_name_1).symbol.values

    profitable_strategies = {}
    best_strategies = {}

    for stock_symbol in stock_symbols:
        if stock_symbol in existing_symbols:
            print(f'Data already exists for {stock_symbol}')
            continue

        print(f'Getting data for {stock_symbol}')

        is_profitable = []
        total_profits = []
        total_trades = []

        data_bars_params = StockBarsRequest(
                        symbol_or_symbols=stock_symbol,
                        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                        start=start_date,
                        end=end_date
                        )

        # Get data
        try:
            data_bars = data_client.get_stock_bars(data_bars_params)
        except Exception as e:
            print(f'Error: {e}')
            print(f'Data unavailable for {stock_symbol}')
            continue

        df = data_bars.df
        df = df.reset_index() # separates symbol and timestamp as columns rather than as multiindex

        # Make sure dataframe only has data for timestamps between 9:30am and 4pm EST
        df = df[(df['timestamp'].dt.tz_convert(est).dt.time >= start_of_trading_day) & (df['timestamp'].dt.tz_convert(est).dt.time <= end_of_trading_day)]
        df.index = df['timestamp']

        # Check if data starts within a month of start_date, otherwise may not be enough data to use this stock
        if df.iloc[0]['timestamp'].date() > start_date.date() + timedelta(days=30):
            print(f'Not enough data present for {stock_symbol}')
            print(f'First date in data: {df.iloc[0]["timestamp"].date()}')
            print(f'Start date: {start_date}')
            continue

        df = adjust_for_stock_split(df, stock_symbol, year)

        hourly_df = pd.DataFrame()
        hourly_df.index = df.groupby(df.index.tz_convert(est).hour).all().index
        hourly_df['symbol'] = [stock_symbol] * len(hourly_df.index)
        hourly_df['hour'] = hourly_df.index

        # Baseline- purely time based
        baseline_buy_signal, baseline_sell_signal = get_baseline_signals(df)

        # SMA Crossover
        crossover_signal = get_sma_crossover_signal(df, 20, 50)
        hourly_mean_crossover_signal = get_hourly_sma_crossover_signal(df, 20, 50)

        # Slow Stochastic Oscillator
        slow_stochastic_oscillator = get_slow_stochastic_oscillator(df, '14D', '3D', 20, 80)
        slow_stochastic_oscillator_hourly = get_hourly_slow_stochastic_oscillator(df, '14D', '3D', 20, 80)

        # Mean Reversion Strategy
        mean_reversion_signal = get_mean_reversion_signal(df, '20D', [-1.5, 1.5])
        mean_reversion_signal_hourly = get_hourly_mean_reversion_signal(df, '20D', [-1.5, 1.5])

        # RSI
        rsi_signal = get_rsi_signal(df, '3D', 30, 70)
        rsi_signal_hourly = get_hourly_rsi_signal(df, '14D', 30, 70)

        # Compute profits from all methods
        baseline_buy_signal = format_trade_signals(baseline_buy_signal, True)
        baseline_sell_signal = format_trade_signals(baseline_sell_signal, True)
        total_profits_baseline, percent_change_baseline = determine_profits(baseline_buy_signal["close"], baseline_sell_signal["close"])
        is_profitable.append(int(total_profits_baseline > 0))
        total_profits.append(total_profits_baseline)
        total_trades.append(get_total_trades(baseline_buy_signal, True) + get_total_trades(baseline_sell_signal, True))
        hourly_profits = get_total_profits_per_hour(baseline_buy_signal, baseline_sell_signal, True)
        hourly_df['baseline_profits'] = hourly_profits
        hourly_df['baseline_is_prof'] = (hourly_profits > 0).astype(int)
        trade_counts = get_total_trades_per_hour(baseline_buy_signal, baseline_sell_signal, True)
        hourly_df['baseline_total_trades'] = trade_counts

        total_profits_crossover = format_data(format_trade_signals(crossover_signal), is_profitable, total_profits, total_trades, hourly_df, 'sma')
        total_profits_crossover_hourly = format_data(format_trade_signals(hourly_mean_crossover_signal), is_profitable, total_profits, total_trades, hourly_df, 'sma_hourly')
        total_profits_stoch = format_data(format_trade_signals(slow_stochastic_oscillator), is_profitable, total_profits, total_trades, hourly_df, 'stoch')
        total_profits_stoch_hourly = format_data(format_trade_signals(slow_stochastic_oscillator_hourly), is_profitable, total_profits, total_trades, hourly_df, 'stoch_hourly')
        total_profits_mean_reversion = format_data(format_trade_signals(mean_reversion_signal), is_profitable, total_profits, total_trades, hourly_df, 'mean_rever')
        total_profits_mean_reversion_hourly = format_data(format_trade_signals(mean_reversion_signal_hourly), is_profitable, total_profits, total_trades, hourly_df, 'mean_rever_hourly')
        total_profits_rsi = format_data(format_trade_signals(rsi_signal), is_profitable, total_profits, total_trades, hourly_df, 'rsi')
        total_profits_rsi_hourly = format_data(format_trade_signals(rsi_signal_hourly), is_profitable, total_profits, total_trades, hourly_df, 'rsi_hourly')

        profitable_strategies[stock_symbol] = is_profitable
        strategies = {'baseline': total_profits_baseline, 'sma_crossover': total_profits_crossover, 'hourly_crossover': total_profits_crossover_hourly, 'stoch_osc': total_profits_stoch, 'stoch_osc_hourly': total_profits_stoch_hourly, 'mean_reversion': total_profits_mean_reversion, 'mean_reversion_hourly': total_profits_mean_reversion_hourly, 'rsi': total_profits_rsi, 'rsi_hourly': total_profits_rsi_hourly}
        best_strategy = max(strategies, key=strategies.get)
        best_strategies[stock_symbol] = best_strategy

        if not Path(file_name_1).exists():
            with open(file_name_1, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(col_names) # + company_data.columns.tolist()

        if not Path(file_name_2).exists():
            with open(file_name_2, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(col_names_hourly) # + company_data.columns.tolist()

        # Summary Statistics
        mean_price = df.mean()['close']
        std_dev = df.std()['close']
        low = df['low'].min()
        high = df['high'].max()

        # log_returns =  np.log(df['close'].iloc[1:] / df['close'].iloc[1:].shift(1))
        # volatility = log_returns.std()

        # Hourly Summary Statistics
        hourly_df['mean_price'] = df.groupby(df.index.tz_convert(est).hour).mean()['close']
        hourly_df['std_dev'] = df.groupby(df.index.tz_convert(est).hour).std()['close']
        hourly_df['low'] = df.groupby(df.index.tz_convert(est).hour)['low'].min()
        hourly_df['high'] = df.groupby(df.index.tz_convert(est).hour)['high'].max()

        # get best strategy per hour
        cols_to_include = ['baseline_profits', 'sma_profits', 'sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits', 'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits']
        hourly_df['best_strategy'] = hourly_df[cols_to_include].idxmax(axis='columns')

        # save is_profitable, profits, total trades, best strategy
        with open(file_name_1, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([stock_symbol] + is_profitable + total_profits + total_trades + [mean_price] + [std_dev] + [low] + [high]+ [best_strategy]) # + company_data.values.tolist()[0]

        hourly_df = hourly_df[col_names_hourly] # +company_data.columns.tolist() .fillna(0)
        # hourly_df = hourly_df.fillna(0)
        hourly_df.to_csv(file_name_2, mode='a', header=False, index=False)

    return profitable_strategies, best_strategies

def get_and_save_raw_data(stock_symbols, data_client, start_date, end_date, file_name):
    start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
    end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST
    est = pytz.timezone('US/Eastern')
    year = start_date.year

    if not Path(file_name).exists():
        existing_symbols = []
    else:
        existing_symbols = pd.read_csv(file_name).symbol.values

    for stock_symbol in stock_symbols:
        if stock_symbol in existing_symbols:
            print(f'Data already exists for {stock_symbol}')
            continue

        print(f'Getting data for {stock_symbol}')

        data_bars_params = StockBarsRequest(
                        symbol_or_symbols=stock_symbol,
                        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                        start=start_date,
                        end=end_date
                        )

        # Get data
        try:
            data_bars = data_client.get_stock_bars(data_bars_params)
        except Exception as e:
            print(f'Error: {e}')
            print(f'Data unavailable for {stock_symbol}')
            continue

        df = data_bars.df
        df = df.reset_index() # separates symbol and timestamp as columns rather than as multiindex

        # Make sure dataframe only has data for timestamps between 9:30am and 4pm EST
        df = df[(df['timestamp'].dt.tz_convert(est).dt.time >= start_of_trading_day) & (df['timestamp'].dt.tz_convert(est).dt.time <= end_of_trading_day)]
        df.index = df['timestamp']

        # Check if data starts within a month of start_date, otherwise may not be enough data to use this stock
        if df.iloc[0]['timestamp'].date() > start_date.date() + timedelta(days=30):
            print(f'Not enough data present for {stock_symbol}')
            print(f'First date in data: {df.iloc[0]["timestamp"].date()}')
            print(f'Start date: {start_date}')
            continue

        df = adjust_for_stock_split(df, stock_symbol, year)

        if not Path(file_name).exists():
            with open(file_name, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(df.columns.tolist())

        df.to_csv(file_name, mode='a', header=False, index=False)

def generate_training_data(df, stock_symbol, strategy_parameters, file_name_1, file_name_2, col_names, col_names_hourly):

    if not Path(file_name_1).exists():
        existing_symbols = []
    else:
        existing_symbols = pd.read_csv(file_name_1).symbol.values

    if stock_symbol in existing_symbols:
        print(f'Training data already saved for {stock_symbol}')
        return

    is_profitable = []
    total_profits = []
    total_trades = []
    profitable_strategies = {}
    best_strategies = {}

    est = pytz.timezone('US/Eastern')

    hourly_df = pd.DataFrame()
    hourly_df.index = df.groupby(df.index.tz_convert(est).hour).all().index
    hourly_df['symbol'] = [stock_symbol] * len(hourly_df.index)
    hourly_df['hour'] = hourly_df.index

    # Baseline- purely time based
    baseline_buy_signal, baseline_sell_signal = get_baseline_signals(df)

    # SMA Crossover
    crossover_signal = get_sma_crossover_signal(df, strategy_parameters['sma']['short_time_period'], strategy_parameters['sma']['long_time_period'])
    hourly_mean_crossover_signal = get_hourly_sma_crossover_signal(df, strategy_parameters['sma_hourly']['short_time_period'], strategy_parameters['sma_hourly']['long_time_period'])

    # Slow Stochastic Oscillator
    slow_stochastic_oscillator = get_slow_stochastic_oscillator(df, f'{strategy_parameters["stoch"]["long_time_period"]}D', f'{strategy_parameters["stoch"]["short_time_period"]}D', 20, 80)
    slow_stochastic_oscillator_hourly = get_hourly_slow_stochastic_oscillator(df, f'{strategy_parameters["stoch_hourly"]["long_time_period"]}D', f'{strategy_parameters["stoch_hourly"]["short_time_period"]}D', 20, 80)

    # Mean Reversion Strategy
    mean_reversion_signal = get_mean_reversion_signal(df, f'{strategy_parameters["mean_rever"]["time_period"]}D', [-1.5, 1.5])
    mean_reversion_signal_hourly = get_hourly_mean_reversion_signal(df, f'{strategy_parameters["mean_rever_hourly"]["time_period"]}D', [-1.5, 1.5])

    # RSI
    rsi_signal = get_rsi_signal(df, f'{strategy_parameters["rsi"]["time_period"]}D', 30, 70)
    rsi_signal_hourly = get_hourly_rsi_signal(df, f'{strategy_parameters["rsi_hourly"]["time_period"]}D', 30, 70)

    # Compute profits from all methods
    baseline_buy_signal = format_trade_signals(baseline_buy_signal, True)
    baseline_sell_signal = format_trade_signals(baseline_sell_signal, True)
    total_profits_baseline, percent_change_baseline = determine_profits(baseline_buy_signal["close"], baseline_sell_signal["close"])
    is_profitable.append(int(total_profits_baseline > 0))
    total_profits.append(total_profits_baseline)
    total_trades.append(get_total_trades(baseline_buy_signal, True) + get_total_trades(baseline_sell_signal, True))
    hourly_profits = get_total_profits_per_hour(baseline_buy_signal, baseline_sell_signal, True)
    hourly_df['baseline_profits'] = hourly_profits
    hourly_df['baseline_is_prof'] = (hourly_profits > 0).astype(int)
    trade_counts = get_total_trades_per_hour(baseline_buy_signal, baseline_sell_signal, True)
    hourly_df['baseline_total_trades'] = trade_counts

    total_profits_crossover = format_data(format_trade_signals(crossover_signal), is_profitable, total_profits, total_trades, hourly_df, 'sma')
    total_profits_crossover_hourly = format_data(format_trade_signals(hourly_mean_crossover_signal), is_profitable, total_profits, total_trades, hourly_df, 'sma_hourly')
    total_profits_stoch = format_data(format_trade_signals(slow_stochastic_oscillator), is_profitable, total_profits, total_trades, hourly_df, 'stoch')
    total_profits_stoch_hourly = format_data(format_trade_signals(slow_stochastic_oscillator_hourly), is_profitable, total_profits, total_trades, hourly_df, 'stoch_hourly')
    total_profits_mean_reversion = format_data(format_trade_signals(mean_reversion_signal), is_profitable, total_profits, total_trades, hourly_df, 'mean_rever')
    total_profits_mean_reversion_hourly = format_data(format_trade_signals(mean_reversion_signal_hourly), is_profitable, total_profits, total_trades, hourly_df, 'mean_rever_hourly')
    total_profits_rsi = format_data(format_trade_signals(rsi_signal), is_profitable, total_profits, total_trades, hourly_df, 'rsi')
    total_profits_rsi_hourly = format_data(format_trade_signals(rsi_signal_hourly), is_profitable, total_profits, total_trades, hourly_df, 'rsi_hourly')

    profitable_strategies[stock_symbol] = is_profitable
    strategies = {'baseline': total_profits_baseline, 'sma_crossover': total_profits_crossover, 'hourly_crossover': total_profits_crossover_hourly, 'stoch_osc': total_profits_stoch, 'stoch_osc_hourly': total_profits_stoch_hourly, 'mean_reversion': total_profits_mean_reversion, 'mean_reversion_hourly': total_profits_mean_reversion_hourly, 'rsi': total_profits_rsi, 'rsi_hourly': total_profits_rsi_hourly}
    best_strategy = max(strategies, key=strategies.get)
    best_strategies[stock_symbol] = best_strategy

    if not Path(file_name_1).exists():
        with open(file_name_1, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(col_names) # + company_data.columns.tolist()

    if not Path(file_name_2).exists():
        with open(file_name_2, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(col_names_hourly) # + company_data.columns.tolist()

    # Summary Statistics
    mean_price = df.mean()['close']
    std_dev = df.std()['close']
    low = df['low'].min()
    high = df['high'].max()

    # log_returns =  np.log(df['close'].iloc[1:] / df['close'].iloc[1:].shift(1))
    # volatility = log_returns.std()

    # Hourly Summary Statistics
    hourly_df['mean_price'] = df.groupby(df.index.tz_convert(est).hour).mean()['close']
    hourly_df['std_dev'] = df.groupby(df.index.tz_convert(est).hour).std()['close']
    hourly_df['low'] = df.groupby(df.index.tz_convert(est).hour)['low'].min()
    hourly_df['high'] = df.groupby(df.index.tz_convert(est).hour)['high'].max()

    # get best strategy per hour
    cols_to_include = ['baseline_profits', 'sma_profits', 'sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits', 'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits']
    hourly_df['best_strategy'] = hourly_df[cols_to_include].idxmax(axis='columns')

    # save is_profitable, profits, total trades, best strategy
    with open(file_name_1, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([stock_symbol] + is_profitable + total_profits + total_trades + [mean_price] + [std_dev] + [low] + [high]+ [best_strategy]) # + company_data.values.tolist()[0]

    hourly_df = hourly_df[col_names_hourly] # +company_data.columns.tolist() .fillna(0)
    # hourly_df = hourly_df.fillna(0)
    hourly_df.to_csv(file_name_2, mode='a', header=False, index=False)

    return profitable_strategies, best_strategies
