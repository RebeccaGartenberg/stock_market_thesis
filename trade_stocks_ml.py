from alpaca.data.historical import StockHistoricalDataClient
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import yfinance as yf
from datetime import datetime, timezone, date, timedelta
from generate_training_data_ml import get_stock_symbols, generate_and_save_training_data, format_training_data, get_company_data, save_company_data, get_income_statement_data
import requests
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn import svm
import time
import csv
import pytz
from pathlib import Path
from statistics import mean
import dataframe_image as dfi

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
AV_API_KEY = params.get("alpha_vantage").get("api_key")
year = params.get("year")
training_data_dir = params.get("training_data_directory")
tables_dir_name = params.get("ml_tables_directory")
stock_list_dir = params.get("stock_list_directory")

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
            'baseline_total_trades','sma_total_trades','sma_hourly_total_trades', 'stoch_total_trades', 'stoch_hourly_total_trades',
            'mean_rever_total_trades', 'mean_rever_hourly_total_trades', 'rsi_total_trades', 'rsi_hourly_total_trades',
            'mean_price','std_dev', 'low', 'high', 'best_strategy'
])

col_names_hourly=(['symbol','hour','baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof',
            'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof',
            'baseline_profits','sma_profits','sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits',
            'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits',
            'baseline_total_trades','sma_total_trades','sma_hourly_total_trades', 'stoch_total_trades', 'stoch_hourly_total_trades',
            'mean_rever_total_trades', 'mean_rever_hourly_total_trades', 'rsi_total_trades', 'rsi_hourly_total_trades',
            'mean_price', 'std_dev', 'low', 'high', 'best_strategy'
])

# symbol_df = pd.read_csv('./stock_lists/stock_symbols.csv', error_bad_lines=False)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
est = pytz.timezone('US/Eastern')

t0_start_date = datetime(year-1, 7, 1)
t0_end_date = datetime(year-1, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t1_start_date = datetime(year, 1, 1)
t1_end_date = datetime(year, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t2_start_date = datetime(year, 7, 1)
t2_end_date = datetime(year, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t3_start_date = datetime(year+1, 1, 1)
t3_end_date = datetime(year+1, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)

# Get symbols for training data
nasdaq_symbols = get_stock_symbols(100, f'{stock_list_dir}/custom-nasdaq-stocks-stocks-all.csv')
nyse_symbols = get_stock_symbols(100, f'{stock_list_dir}/custom-nyse-stocks-stocks-all.csv')
save_company_data(pd.concat([nasdaq_symbols, nyse_symbols]), company_data_file)

stock_symbols = pd.read_csv(company_data_file)["Symbol"].tolist()

# Collect data on symbols
get_income_statement_data(stock_symbols, annual_report_file, quarterly_report_file, AV_API_KEY)

print(f'Getting data for t0: {t0_start_date} to {t0_end_date}')

generate_and_save_training_data(stock_symbols,
                                data_client,
                                t0_start_date,
                                t0_end_date,
                                file_name_t0,
                                file_name_hourly_t0,
                                col_names,
                                col_names_hourly,
                                AV_API_KEY
                                )

print(f'Getting data for t1: {t1_start_date} to {t1_end_date}')

generate_and_save_training_data(stock_symbols,
                                data_client,
                                t1_start_date,
                                t1_end_date,
                                file_name_t1,
                                file_name_hourly_t1,
                                col_names,
                                col_names_hourly,
                                AV_API_KEY
                                )

print(f'Getting data for t2: {t2_start_date} to {t2_end_date}')

generate_and_save_training_data(stock_symbols,
                                data_client,
                                t2_start_date,
                                t2_end_date,
                                file_name_t2,
                                file_name_hourly_t2,
                                col_names,
                                col_names_hourly,
                                AV_API_KEY
                                )

print(f'Getting data for t3: {t3_start_date} to {t3_end_date}')

generate_and_save_training_data(stock_symbols,
                                data_client,
                                t3_start_date,
                                t3_end_date,
                                file_name_t3,
                                file_name_hourly_t3,
                                col_names,
                                col_names_hourly,
                                AV_API_KEY
                                )

labels = ['baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof',
            'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof', 'best_strategy']

# Company data and reports
company_data = pd.read_csv(company_data_file)
annual_reports = pd.read_csv(annual_report_file)
quarterly_reports = pd.read_csv(quarterly_report_file)

# profitable strategies per stock
profitable_strategies_t0 = pd.read_csv(file_name_t0)
profitable_strategies_t1 = pd.read_csv(file_name_t1)
profitable_strategies_t2 = pd.read_csv(file_name_t2)
profitable_strategies_t3 = pd.read_csv(file_name_t3)

# profitable strategies per hour per stock
profitable_strategies_hourly_t0 = pd.read_csv(file_name_hourly_t0)
profitable_strategies_hourly_t1 = pd.read_csv(file_name_hourly_t1)
profitable_strategies_hourly_t2 = pd.read_csv(file_name_hourly_t2)
profitable_strategies_hourly_t3 = pd.read_csv(file_name_hourly_t3)

quarterly_reports_df = quarterly_reports[['fiscalDateEnding', 'symbol', 'totalRevenue', 'ebitda', 'ebit']]
quarterly_reports_df[['totalRevenue', 'ebitda', 'ebit']] = quarterly_reports_df[['totalRevenue', 'ebitda', 'ebit']].apply(pd.to_numeric, errors='coerce')

quarterly_reports_df[(pd.to_datetime(quarterly_reports_df["fiscalDateEnding"]) > pd.to_datetime(t0_start_date)) & (pd.to_datetime(quarterly_reports_df["fiscalDateEnding"]) <= pd.to_datetime(t0_end_date.date())) & (quarterly_reports_df['symbol'] == 'ZIVO')]

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

training_data_hourly = training_data_hourly_t1[['hour']+labels].merge(training_data_hourly_t0.drop(labels, axis=1), how='left', right_on=['symbol','hour'], left_on=['symbol','hour'])
testing_data_hourly = training_data_hourly_t2[['hour']+labels].merge(training_data_hourly_t1.drop(labels, axis=1), how='left', right_on=['symbol','hour'], left_on=['symbol','hour'])

imputer = SimpleImputer(strategy='mean')

# Features
X_train = imputer.fit_transform(training_data.drop(columns=labels))
X_test = imputer.fit_transform(testing_data.drop(columns=labels))

# Labels
Y_train = training_data[labels]
Y_test = testing_data[labels]

regression_models = []
random_forest_models = []
svm_models = []
for i in range(0, Y_train.shape[1]):
    regression_model = LogisticRegression(max_iter=1000)
    regression_model.fit(X_train, Y_train[labels[i]])
    regression_models.append(regression_model)

    random_forest_model = RandomForestClassifier(n_estimators = 100)
    random_forest_model.fit(X_train, Y_train[labels[i]])
    random_forest_models.append(random_forest_model)

    svm_model = svm.SVC()
    svm_model.fit(X_train, Y_train[labels[i]])
    svm_models.append(svm_model)

# Make predictions for each label on the test data
y_pred_regression = np.column_stack([regression_model.predict(X_test) for regression_model in regression_models])
y_pred_random_forest = np.column_stack([random_forest_model.predict(X_test) for random_forest_model in random_forest_models])
y_pred_svm = np.column_stack([svm_model.predict(X_test) for svm_model in svm_models])

accuracies_regression_is_profitable = [accuracy_score(Y_test.to_numpy()[:,i].tolist(), y_pred_regression[:,i].tolist()) for i in range(Y_test.shape[1]-1)]
accuracies_regression_best_strategy = accuracy_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_regression[:,Y_test.shape[1]-1])
precisions_regression_is_profitable = [precision_score(Y_test.to_numpy()[:,i].tolist(), y_pred_regression[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
precisions_regression_best_strategy = precision_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_regression[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
recalls_regression_is_profitable = [recall_score(Y_test.to_numpy()[:,i].tolist(), y_pred_regression[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
recalls_regression_best_strategy = recall_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_regression[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
f1_regression_is_profitable = [f1_score(Y_test.to_numpy()[:,i].tolist(), y_pred_regression[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
f1_regression_best_strategy = f1_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_regression[:,Y_test.shape[1]-1], average='macro', zero_division='warn')

accuracies_random_forest_is_profitable = [accuracy_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist()) for i in range(Y_test.shape[1]-1)]
accuracies_random_forest_best_strategy = accuracy_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1])
precisions_random_forest_is_profitable = [precision_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
precisions_random_forest_best_strategy = precision_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
recalls_random_forest_is_profitable = [recall_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
recalls_random_forest_best_strategy = recall_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
f1_random_forest_is_profitable = [f1_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
f1_random_forest_best_strategy = f1_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')

accuracies_svm_is_profitable = [accuracy_score(Y_test.to_numpy()[:,i].tolist(), y_pred_svm[:,i].tolist()) for i in range(Y_test.shape[1]-1)]
accuracies_svm_best_strategy = accuracy_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_svm[:,Y_test.shape[1]-1])
precisions_svm_is_profitable = [precision_score(Y_test.to_numpy()[:,i].tolist(), y_pred_svm[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
precisions_svm_best_strategy = precision_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_svm[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
recalls_svm_is_profitable = [recall_score(Y_test.to_numpy()[:,i].tolist(), y_pred_svm[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
recalls_svm_best_strategy = recall_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_svm[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
f1_svm_is_profitable = [f1_score(Y_test.to_numpy()[:,i].tolist(), y_pred_svm[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
f1_svm_best_strategy = f1_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_svm[:,Y_test.shape[1]-1], average='macro', zero_division='warn')

results = ({
        'Metric' : ['Accuracy', 'Precision', 'Recall', 'F1'],
        'LR is_profitable': [mean(accuracies_regression_is_profitable), mean(precisions_regression_is_profitable),  mean(recalls_regression_is_profitable), mean(f1_regression_is_profitable)],
        'RF is_profitable': [mean(accuracies_random_forest_is_profitable), mean(precisions_random_forest_is_profitable), mean(recalls_random_forest_is_profitable), mean(f1_random_forest_is_profitable)],
        'SVM is_profitable': [mean(accuracies_svm_is_profitable), mean(precisions_svm_is_profitable), mean(recalls_svm_is_profitable), mean(f1_svm_is_profitable)],
        'LR best_strategy': [accuracies_regression_best_strategy, precisions_regression_best_strategy, recalls_regression_best_strategy, f1_regression_best_strategy],
        'RF best_strategy': [accuracies_random_forest_best_strategy, precisions_random_forest_best_strategy, recalls_random_forest_best_strategy, f1_random_forest_best_strategy],
        'SVM best_strategy': [accuracies_svm_best_strategy, precisions_svm_best_strategy, recalls_svm_best_strategy, f1_svm_best_strategy],
        })

results_table = pd.DataFrame(results).set_index('Metric')
with open(f'{tables_dir_name}/ml_results_table_1.tex','w') as tf:
    tf.write(results_table.to_latex())
dfi.export(results_table.style, f'{tables_dir_name}/ml_results_table_1.png')

imputer = SimpleImputer(strategy='mean')

# Features
X_train = imputer.fit_transform(training_data_hourly.drop(columns=labels))
X_test = imputer.fit_transform(testing_data_hourly.drop(columns=labels))

# Labels
Y_train = training_data_hourly[labels].fillna(0)
Y_test = testing_data_hourly[labels].fillna(0)

regression_models = []
random_forest_models = []
svm_models = []
for i in range(0, Y_train.shape[1]):
    regression_model = LogisticRegression(max_iter=1000)
    regression_model.fit(X_train, Y_train[labels[i]])
    regression_models.append(regression_model)

    random_forest_model = RandomForestClassifier(n_estimators = 100)
    random_forest_model.fit(X_train, Y_train[labels[i]])
    random_forest_models.append(random_forest_model)

    svm_model = svm.SVC()
    svm_model.fit(X_train, Y_train[labels[i]])
    svm_models.append(svm_model)

# Make predictions for each label on the test data
y_pred_regression = np.column_stack([regression_model.predict(X_test) for regression_model in regression_models])
y_pred_random_forest = np.column_stack([random_forest_model.predict(X_test) for random_forest_model in random_forest_models])
y_pred_svm = np.column_stack([svm_model.predict(X_test) for svm_model in svm_models])

accuracies_regression_is_profitable = [accuracy_score(Y_test.to_numpy()[:,i].tolist(), y_pred_regression[:,i].tolist()) for i in range(Y_test.shape[1]-1)]
accuracies_regression_best_strategy = accuracy_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_regression[:,Y_test.shape[1]-1])
precisions_regression_is_profitable = [precision_score(Y_test.to_numpy()[:,i].tolist(), y_pred_regression[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
precisions_regression_best_strategy = precision_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_regression[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
recalls_regression_is_profitable = [recall_score(Y_test.to_numpy()[:,i].tolist(), y_pred_regression[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
recalls_regression_best_strategy = recall_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_regression[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
f1_regression_is_profitable = [f1_score(Y_test.to_numpy()[:,i].tolist(), y_pred_regression[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
f1_regression_best_strategy = f1_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_regression[:,Y_test.shape[1]-1], average='macro', zero_division='warn')

accuracies_random_forest_is_profitable = [accuracy_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist()) for i in range(Y_test.shape[1]-1)]
accuracies_random_forest_best_strategy = accuracy_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1])
precisions_random_forest_is_profitable = [precision_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
precisions_random_forest_best_strategy = precision_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
recalls_random_forest_is_profitable = [recall_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
recalls_random_forest_best_strategy = recall_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
f1_random_forest_is_profitable = [f1_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
f1_random_forest_best_strategy = f1_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')

accuracies_svm_is_profitable = [accuracy_score(Y_test.to_numpy()[:,i].tolist(), y_pred_svm[:,i].tolist()) for i in range(Y_test.shape[1]-1)]
accuracies_svm_best_strategy = accuracy_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_svm[:,Y_test.shape[1]-1])
precisions_svm_is_profitable = [precision_score(Y_test.to_numpy()[:,i].tolist(), y_pred_svm[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
precisions_svm_best_strategy = precision_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_svm[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
recalls_svm_is_profitable = [recall_score(Y_test.to_numpy()[:,i].tolist(), y_pred_svm[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
recalls_svm_best_strategy = recall_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_svm[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
f1_svm_is_profitable = [f1_score(Y_test.to_numpy()[:,i].tolist(), y_pred_svm[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
f1_svm_best_strategy = f1_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_svm[:,Y_test.shape[1]-1], average='macro', zero_division='warn')

results = ({
        'Metric' : ['Accuracy', 'Precision', 'Recall', 'F1'],
        'LR is_profitable': [mean(accuracies_regression_is_profitable), mean(precisions_regression_is_profitable),  mean(recalls_regression_is_profitable), mean(f1_regression_is_profitable)],
        'RF is_profitable': [mean(accuracies_random_forest_is_profitable), mean(precisions_random_forest_is_profitable), mean(recalls_random_forest_is_profitable), mean(f1_random_forest_is_profitable)],
        'SVM is_profitable': [mean(accuracies_svm_is_profitable), mean(precisions_svm_is_profitable), mean(recalls_svm_is_profitable), mean(f1_svm_is_profitable)],
        'LR best_strategy': [accuracies_regression_best_strategy, precisions_regression_best_strategy, recalls_regression_best_strategy, f1_regression_best_strategy],
        'RF best_strategy': [accuracies_random_forest_best_strategy, precisions_random_forest_best_strategy, recalls_random_forest_best_strategy, f1_random_forest_best_strategy],
        'SVM best_strategy': [accuracies_svm_best_strategy, precisions_svm_best_strategy, recalls_svm_best_strategy, f1_svm_best_strategy],
        })

results_table = pd.DataFrame(results).set_index('Metric')
with open(f'{tables_dir_name}/ml_results_table_2.tex','w') as tf:
    tf.write(results_table.to_latex())
dfi.export(results_table.style, f'{tables_dir_name}/ml_results_table_2.png')
