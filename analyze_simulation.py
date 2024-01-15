import yaml
import pdb
import pandas as pd
from generate_training_data_ml import format_training_data
from datetime import datetime
import pytz
from ml_helper_functions import run_ml_models_simulation, normalize_data

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

training_data_dir = params.get("training_data_directory")
live_sim_training_data_dir = params.get("live_sim_training_data_directory")
live_sim_results_dir = params.get("live_sim_results_directory")
trade_info_file_name = f'{live_sim_training_data_dir}/trade_info_live_sim.csv'

simulation_results = pd.read_csv(trade_info_file_name)
company_data_file = f'{training_data_dir}/company_data.csv'
annual_report_file = f'{training_data_dir}/annual_reports.csv'
quarterly_report_file = f'{training_data_dir}/quarterly_reports.csv'
file_name_t1 = f'{training_data_dir}/stock_strategy_data_t1.csv'
file_name_t2 = f'{training_data_dir}/stock_strategy_data_t2.csv'
file_name_t3 = f'{training_data_dir}/stock_strategy_data_t3.csv'
file_name_hourly_t1 = f'{training_data_dir}/stock_strategy_data_hourly_t1.csv'
file_name_hourly_t2 = f'{training_data_dir}/stock_strategy_data_hourly_t2.csv'
file_name_hourly_t3 = f'{training_data_dir}/stock_strategy_data_hourly_t3.csv'

est = pytz.timezone('US/Eastern')
year = params.get("year")
t1_start_date = datetime(year, 1, 1)
t1_end_date = datetime(year, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t2_start_date = datetime(year, 7, 1)
t2_end_date = datetime(year, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)
t3_start_date = datetime(year+1, 1, 1)
t3_end_date = datetime(year+1, 6, 30).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)

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
# profitable_strategies_t1 = pd.read_csv(file_name_t1)
# profitable_strategies_t2 = pd.read_csv(file_name_t2)
# profitable_strategies_t3 = pd.read_csv(file_name_t3)
#
# extra_t1 = list(set(profitable_strategies_t1['symbol'].tolist()) - set(profitable_strategies_t3['symbol'].tolist())) + list(set(profitable_strategies_t1['symbol'].tolist()) - set(profitable_strategies_t2['symbol'].tolist()))
# extra_t2 = list(set(profitable_strategies_t2['symbol'].tolist()) - set(profitable_strategies_t3['symbol'].tolist())) + list(set(profitable_strategies_t2['symbol'].tolist()) - set(profitable_strategies_t1['symbol'].tolist()))
# extra_t3 = list(set(profitable_strategies_t3['symbol'].tolist()) - set(profitable_strategies_t1['symbol'].tolist())) + list(set(profitable_strategies_t3['symbol'].tolist()) - set(profitable_strategies_t2['symbol'].tolist()))
#
# profitable_strategies_t1 = profitable_strategies_t1[~profitable_strategies_t1['symbol'].isin(extra_t1)]
# profitable_strategies_t2 = profitable_strategies_t2[~profitable_strategies_t2['symbol'].isin(extra_t2)]
# profitable_strategies_t3 = profitable_strategies_t3[~profitable_strategies_t3['symbol'].isin(extra_t3)]
#
# # profitable strategies per hour per stock
# profitable_strategies_hourly_t1 = pd.read_csv(file_name_hourly_t1)
# profitable_strategies_hourly_t2 = pd.read_csv(file_name_hourly_t2)
# profitable_strategies_hourly_t3 = pd.read_csv(file_name_hourly_t3)
#
# extra_t1_hourly = list(set(profitable_strategies_hourly_t1['symbol'].tolist()) - set(profitable_strategies_hourly_t3['symbol'].tolist())) + list(set(profitable_strategies_hourly_t1['symbol'].tolist()) - set(profitable_strategies_hourly_t2['symbol'].tolist()))
# extra_t2_hourly = list(set(profitable_strategies_hourly_t2['symbol'].tolist()) - set(profitable_strategies_hourly_t3['symbol'].tolist())) + list(set(profitable_strategies_hourly_t2['symbol'].tolist()) - set(profitable_strategies_hourly_t1['symbol'].tolist()))
# extra_t3_hourly = list(set(profitable_strategies_hourly_t3['symbol'].tolist()) - set(profitable_strategies_hourly_t1['symbol'].tolist())) + list(set(profitable_strategies_hourly_t3['symbol'].tolist()) - set(profitable_strategies_hourly_t2['symbol'].tolist()))
#
# profitable_strategies_hourly_t1 = profitable_strategies_hourly_t1[~profitable_strategies_hourly_t1['symbol'].isin(extra_t1_hourly)]
# profitable_strategies_hourly_t2 = profitable_strategies_hourly_t2[~profitable_strategies_hourly_t2['symbol'].isin(extra_t2_hourly)]
# profitable_strategies_hourly_t3 = profitable_strategies_hourly_t3[~profitable_strategies_hourly_t3['symbol'].isin(extra_t3_hourly)]
#
# quarterly_reports_df = quarterly_reports[['fiscalDateEnding', 'symbol', 'totalRevenue', 'ebitda', 'ebit']]
# quarterly_reports_df[['totalRevenue', 'ebitda', 'ebit']] = quarterly_reports_df[['totalRevenue', 'ebitda', 'ebit']].apply(pd.to_numeric, errors='coerce')
#
# # select columns to use from company_data and quarterly_reports, combine with profitable strategies
# training_data_t1 = format_training_data(profitable_strategies_t1, company_data, quarterly_reports_df, t1_start_date, t1_end_date)
# training_data_t2 = format_training_data(profitable_strategies_t2, company_data, quarterly_reports_df, t2_start_date, t2_end_date)
# training_data_t3 = format_training_data(profitable_strategies_t3, company_data, quarterly_reports_df, t3_start_date, t3_end_date)
#
# training_data_hourly_t1 = format_training_data(profitable_strategies_hourly_t1, company_data, quarterly_reports_df, t1_start_date, t1_end_date)
# training_data_hourly_t2 = format_training_data(profitable_strategies_hourly_t2, company_data, quarterly_reports_df, t2_start_date, t2_end_date)
# training_data_hourly_t3 = format_training_data(profitable_strategies_hourly_t3, company_data, quarterly_reports_df, t3_start_date, t3_end_date)
#
# training_data_t1.set_index('symbol', inplace=True)
# training_data_t2.set_index('symbol', inplace=True)
# training_data_t3.set_index('symbol', inplace=True)
#
# training_data_hourly_t1.set_index('symbol', inplace=True)
# training_data_hourly_t2.set_index('symbol', inplace=True)
# training_data_hourly_t3.set_index('symbol', inplace=True)
#
# training_data = training_data_t2[labels].merge(training_data_t1.drop(labels, axis=1), how='left', right_on='symbol', left_on='symbol')
# testing_data = training_data_t3[labels].merge(training_data_t2.drop(labels, axis=1), how='left', right_on='symbol', left_on='symbol')
#
# training_data_hourly = training_data_hourly_t2[['hour']+labels_hourly].merge(training_data_hourly_t1.drop(labels_hourly, axis=1), how='left', right_on=['symbol','hour'], left_on=['symbol','hour'])
# testing_data_hourly = training_data_hourly_t3[['hour']+labels_hourly].merge(training_data_hourly_t2.drop(labels_hourly, axis=1), how='left', right_on=['symbol','hour'], left_on=['symbol','hour'])
#
# # Normalize and scale features
# normalized_training_data = normalize_data(training_data, labels)
# normalized_testing_data = normalize_data(testing_data, labels)
# normalized_training_data_hourly = normalize_data(training_data_hourly, labels_hourly)
# normalized_testing_data_hourly = normalize_data(testing_data_hourly, labels_hourly)
#
stock_symbols = ['TSLA', 'TM', 'SBUX', 'AMZN', 'MSFT', 'ORCL', 'AMAT', 'TSM', 'MRNA', 'JNJ', 'WMT', 'COST', 'UNH', 'CVS', 'JPM', 'PYPL', 'PEP', 'KO']
#
# thresh = 5
# non_zero_count = (normalized_training_data != 0).sum()
# filtered_columns = non_zero_count[non_zero_count > thresh].index
# filtered_normalized_training_data = normalized_training_data[filtered_columns]
#
# non_zero_count = (normalized_testing_data != 0).sum()
# filtered_columns = non_zero_count[non_zero_count > thresh].index
# filtered_normalized_testing_data = normalized_testing_data[filtered_columns]
# # filtered_normalized_testing_data = filtered_normalized_testing_data.loc[stock_symbols]
#
# thresh = 10
# non_zero_count = (normalized_training_data_hourly != 0).sum()
# filtered_columns = non_zero_count[non_zero_count > thresh].index
# filtered_normalized_training_data_hourly = normalized_training_data_hourly[filtered_columns]
#
# non_zero_count = (normalized_testing_data_hourly != 0).sum()
# filtered_columns = non_zero_count[non_zero_count > thresh].index
# filtered_normalized_testing_data_hourly = normalized_testing_data_hourly[filtered_columns]
# # filtered_normalized_testing_data_hourly = filtered_normalized_testing_data_hourly.loc[stock_symbols]
#
# run_ml_models_simulation(filtered_normalized_training_data, filtered_normalized_testing_data, labels, live_sim_results_dir, 'svm')
# run_ml_models_simulation(filtered_normalized_training_data_hourly, filtered_normalized_testing_data_hourly, labels_hourly, live_sim_results_dir, 'regression', 'hourly')
#
grouped_results = simulation_results.groupby(['symbol', 'strategy']).apply(lambda x: x.iloc[[-1]])
grouped_results[grouped_results['total_account_value'] > 1000]
grouped_results['total_account_value'] = round(grouped_results['total_account_value'], 2)
successful_results = grouped_results[grouped_results['total_account_value'] > 1000]['total_account_value'].reset_index().drop('level_2', axis=1)
# show which strategies were most successful, which were least successful, which stocks/industries were successful or not
# show beginning and end price for each stock
successful_results.to_latex(f'{live_sim_results_dir}/successful_sim_results.tex', header=True, index=False)

simulation_results_ml = pd.read_csv(f'{live_sim_results_dir}/y_pred_svm_.csv')
simulation_results_ml_hourly = pd.read_csv(f'{live_sim_results_dir}/y_pred_regression_hourly.csv')
simulation_results_ml.set_index('symbol', inplace=True)
simulation_results_ml_hourly.set_index('symbol', inplace=True)

simulation_results_ml.loc[stock_symbols]
simulation_results_ml_hourly.loc[stock_symbols]

pdb.set_trace()

# group by timestamp
