from alpaca.data.historical import StockHistoricalDataClient
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import yfinance as yf
from datetime import datetime, timezone, date, timedelta
from generate_training_data_ml import get_stock_symbols, generate_and_save_training_data
import requests
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
tables_dir_name = params.get("ml_tables_directory")

file_name = 'stock_strategy_data.csv'
file_name_2 = 'stock_company_data.csv'
file_name_3 = 'stock_strategy_data_hourly.csv'

file_name_t1 = 'stock_strategy_data_t1.csv'
file_name_t2 = 'stock_strategy_data_t2.csv'
file_name_hourly_t1 = 'stock_strategy_data__hourly_t1.csv'
file_name_hourly_t2 = 'stock_strategy_data_hourly_t2.csv'

# Only use these lines first time
# col_names=(['symbol','baseline','sma','sma_hourly', 'stoch', 'stoch_hourly', 'mean_rever', 'mean_rever_hourly', 'rsi', 'rsi_hourly'])
col_names=(['symbol','baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof',
            'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof',
            'baseline_profits','sma_profits','sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits',
            'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits',
            'baseline_total_trades','sma_total_trades','sma_hourly_total_trades', 'stoch_total_trades', 'stoch_hourly_total_trades',
            'mean_rever_total_trades', 'mean_rever_hourly_total_trades', 'rsi_total_trades', 'rsi_hourly_total_trades',
            'mean_price','std_dev','best_strategy'
])
if not Path(file_name_t1).exists():
    with open(file_name_t1, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(col_names)

if not Path(file_name_t2).exists():
    with open(file_name_t2, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(col_names)

# strategy_col_names = (['symbol','baseline','sma','sma_hourly', 'stoch', 'stoch_hourly', 'mean_rever', 'mean_rever_hourly', 'rsi', 'rsi_hourly'])

# Run once to insert column names in csvs

# with open(file_name, 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(col_names)
col_names_hourly=(['symbol','hour','baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof',
            'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof',
            'baseline_profits','sma_profits','sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits',
            'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits',
            'baseline_total_trades','sma_total_trades','sma_hourly_total_trades', 'stoch_total_trades', 'stoch_hourly_total_trades',
            'mean_rever_total_trades', 'mean_rever_hourly_total_trades', 'rsi_total_trades', 'rsi_hourly_total_trades',
            'mean_price', 'std_dev', 'best_strategy'
])
if not Path(file_name_hourly_t1).exists():
    with open(file_name_hourly_t1, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(col_names_hourly)

if not Path(file_name_hourly_t2).exists():
    with open(file_name_hourly_t2, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(col_names_hourly)

stock_symbols = get_stock_symbols(5)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
est = pytz.timezone('US/Eastern')
t1_start_date = datetime(year, 1, 1)
t1_end_date = datetime(year, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t2_start_date = datetime(year, 7, 1)
t2_end_date = datetime(year, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
# Uncomment for custom dates
# data_end_date = (datetime.today()+timedelta(days=-10)).replace(hour=16, minute=0, second=0, microsecond=0, tzinfo = est)
# data_start_date = (data_end_date+timedelta(days=-5)).replace(hour=9, minute=0, second=0, microsecond=0, tzinfo = est)

profitable_strategies, best_strategies = generate_and_save_training_data(
                                            stock_symbols,
                                            data_client,
                                            t1_start_date,
                                            t1_end_date,
                                            file_name_t1,
                                            file_name_hourly_t1,
                                            col_names_hourly
                                        )

profitable_strategies_t1 = pd.read_csv(file_name_t1, error_bad_lines=False)
profitable_strategies_hourly_t1 = pd.read_csv(file_name_hourly_t1, error_bad_lines=False)

profitable_strategies, best_strategies = generate_and_save_training_data(
                                            stock_symbols,
                                            data_client,
                                            t2_start_date,
                                            t2_end_date,
                                            file_name_t2,
                                            file_name_hourly_t2,
                                            col_names_hourly
                                        )
profitable_strategies_t2 = pd.read_csv(file_name_t2, error_bad_lines=False)
profitable_strategies_hourly_t2 = pd.read_csv(file_name_hourly_t2, error_bad_lines=False)

profitable_strategies_t1.set_index('symbol', inplace=True)
profitable_strategies_t2.set_index('symbol', inplace=True)
strategy_result_col_names = ['baseline_is_prof','sma_is_prof','sma_hourly_is_prof', 'stoch_is_prof', 'stoch_hourly_is_prof', 'mean_rever_is_prof', 'mean_rever_hourly_is_prof', 'rsi_is_prof', 'rsi_hourly_is_prof', 'best_strategy']

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(profitable_strategies_t1.drop(columns=strategy_result_col_names))
X_test = imputer.fit_transform(profitable_strategies_t2.drop(columns=strategy_result_col_names))

Y_train = profitable_strategies_t1[strategy_result_col_names]
Y_test = profitable_strategies_t2[strategy_result_col_names]

regression_models = []
random_forest_models = []
for i in range(0, Y_train.shape[1]):
    regression_model = LogisticRegression(max_iter=1000)
    regression_model.fit(X_train, Y_train[strategy_result_col_names[i]])
    regression_models.append(regression_model)

    random_forest_model = RandomForestClassifier(n_estimators = 100)
    random_forest_model.fit(X_train, Y_train[strategy_result_col_names[i]])
    random_forest_models.append(random_forest_model)

# Make predictions for each label on the test data
y_pred_regression = np.column_stack([regression_model.predict(X_test) for regression_model in regression_models])
y_pred_random_forest = np.column_stack([random_forest_model.predict(X_test) for random_forest_model in random_forest_models])

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

results = ({
        'Metric' : ['Accuracy', 'Precision', 'Recall', 'F1'],
        'LR is_profitable': [mean(accuracies_regression_is_profitable), mean(precisions_regression_is_profitable),  mean(recalls_regression_is_profitable), mean(f1_regression_is_profitable)],
        'RF is_profitable': [mean(accuracies_random_forest_is_profitable), mean(precisions_random_forest_is_profitable), mean(recalls_random_forest_is_profitable), mean(f1_random_forest_is_profitable)],
        'LR best_strategy': [accuracies_regression_best_strategy, precisions_regression_best_strategy, recalls_regression_best_strategy, f1_regression_best_strategy],
        'RF best_strategy': [accuracies_random_forest_best_strategy, precisions_random_forest_best_strategy, recalls_random_forest_best_strategy, f1_random_forest_best_strategy]
        })

results_table = pd.DataFrame(results).set_index('Metric')
with open(f'{tables_dir_name}/ml_results_table_1.tex','w') as tf:
    tf.write(results_table.to_latex())
dfi.export(results_table.style, f'{tables_dir_name}/ml_results_table_1.png')

profitable_strategies_hourly_t1.set_index('symbol', inplace=True)
profitable_strategies_hourly_t2.set_index('symbol', inplace=True)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(profitable_strategies_hourly_t1.drop(columns=strategy_result_col_names))
X_test = imputer.fit_transform(profitable_strategies_hourly_t2.drop(columns=strategy_result_col_names))

Y_train = profitable_strategies_hourly_t1[strategy_result_col_names]
Y_test = profitable_strategies_hourly_t2[strategy_result_col_names]

regression_models = []
random_forest_models = []
for i in range(0, Y_train.shape[1]):
    regression_model = LogisticRegression(max_iter=1000)
    regression_model.fit(X_train, Y_train[strategy_result_col_names[i]])
    regression_models.append(regression_model)

    random_forest_model = RandomForestClassifier(n_estimators = 100)
    random_forest_model.fit(X_train, Y_train[strategy_result_col_names[i]])
    random_forest_models.append(random_forest_model)

# Make predictions for each label on the test data
y_pred_regression = np.column_stack([regression_model.predict(X_test) for regression_model in regression_models])
y_pred_random_forest = np.column_stack([random_forest_model.predict(X_test) for random_forest_model in random_forest_models])

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
accuracies_random_forest = [accuracy_score(Y_test.to_numpy().astype(str)[:, i], y_pred_random_forest.astype(str)[:, i]) for i in range(Y_test.shape[1])]
precisions_random_forest_is_profitable = [precision_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
precisions_random_forest_best_strategy = precision_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
recalls_random_forest_is_profitable = [recall_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
recalls_random_forest_best_strategy = recall_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
f1_random_forest_is_profitable = [f1_score(Y_test.to_numpy()[:,i].tolist(), y_pred_random_forest[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
f1_random_forest_best_strategy = f1_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred_random_forest[:,Y_test.shape[1]-1], average='macro', zero_division='warn')

results = ({
        'Metric' : ['Accuracy', 'Precision', 'Recall', 'F1'],
        'LR is_profitable': [mean(accuracies_regression_is_profitable), mean(precisions_regression_is_profitable),  mean(recalls_regression_is_profitable), mean(f1_regression_is_profitable)],
        'RF is_profitable': [mean(accuracies_random_forest_is_profitable), mean(precisions_random_forest_is_profitable), mean(recalls_random_forest_is_profitable), mean(f1_random_forest_is_profitable)],
        'LR best_strategy': [accuracies_regression_best_strategy, precisions_regression_best_strategy, recalls_regression_best_strategy, f1_regression_best_strategy],
        'RF best_strategy': [accuracies_random_forest_best_strategy, precisions_random_forest_best_strategy, recalls_random_forest_best_strategy, f1_random_forest_best_strategy]
        })

results_table = pd.DataFrame(results).set_index('Metric')
with open(f'{tables_dir_name}/ml_results_table_2.tex','w') as tf:
    tf.write(results_table.to_latex())
dfi.export(results_table.style, f'{tables_dir_name}/ml_results_table_2.png')
