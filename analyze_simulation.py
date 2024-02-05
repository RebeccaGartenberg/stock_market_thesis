import yaml
import pandas as pd
from generate_training_data_ml import format_training_data
from datetime import datetime
import pytz
from ml_helper_functions import run_ml_models_simulation, normalize_data
from analyze_trades import determine_profits, get_total_profits_per_hour

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

stock_symbols = ['TSLA', 'TM', 'SBUX', 'AMZN', 'MSFT', 'ORCL', 'AMAT', 'TSM', 'MRNA', 'JNJ', 'WMT', 'COST', 'UNH', 'CVS', 'JPM', 'PYPL', 'PEP', 'KO']

grouped_results = simulation_results.groupby(['symbol', 'strategy']).apply(lambda x: x.iloc[[-1]])
grouped_results[grouped_results['total_account_value'] > 1000]
grouped_results['total_account_value'] = round(grouped_results['total_account_value'], 2)
successful_results = grouped_results[grouped_results['total_account_value'] > 1000]['total_account_value'].reset_index().drop('level_2', axis=1)
successful_results.to_latex(f'{live_sim_results_dir}/successful_sim_results.tex', header=True, index=False)

simulation_results_ml = pd.read_csv(f'{live_sim_results_dir}/y_pred_svm_.csv')
simulation_results_ml_hourly = pd.read_csv(f'{live_sim_results_dir}/y_pred_regression_hourly.csv')
simulation_results_ml.set_index('symbol', inplace=True)
simulation_results_ml_hourly.set_index('symbol', inplace=True)

simulation_results_ml.loc[stock_symbols]
simulation_results_ml_hourly.loc[stock_symbols]

simulation_results['hour'] = pd.to_datetime(simulation_results['timestamp']).apply(lambda x: x.hour)
grouped_results_2 = simulation_results.groupby(['symbol', 'strategy']).apply(lambda x: x)

symbols = simulation_results.symbol.unique().tolist()
strategies = simulation_results.strategy.unique().tolist()

hourly_results = pd.DataFrame(columns=['symbol', 'strategy', 'hour', 'gain/loss'])
hourly_results_buying = pd.DataFrame(columns=['symbol', 'strategy', 'hour', 'gain/loss'])

for symbol in symbols:
    for strategy in strategies:
        strategy_symbol_results = simulation_results[(simulation_results['symbol'] == symbol) & (simulation_results['strategy']==strategy)]
        buy_signal = strategy_symbol_results[strategy_symbol_results['buy/sell']=='buy']
        sell_signal = strategy_symbol_results[strategy_symbol_results['buy/sell']=='sell']

        avg_gain_loss = strategy_symbol_results.groupby('hour').mean()['gain/loss'].reset_index()[['hour', 'gain/loss']]
        avg_gain_loss['symbol'] = symbol
        avg_gain_loss['strategy'] = strategy
        hourly_results = pd.concat([hourly_results, avg_gain_loss])

        avg_gain_loss_buy = strategy_symbol_results[['hour', 'gain/loss']].set_index('hour').shift(-1).groupby('hour').mean().reset_index()[['hour', 'gain/loss']]
        avg_gain_loss_buy['symbol'] = symbol
        avg_gain_loss_buy['strategy'] = strategy
        hourly_results_buying = pd.concat([hourly_results_buying, avg_gain_loss_buy])

hourly_results.to_latex(f'{live_sim_results_dir}/hourly_results.tex', header=True, index=False)
hourly_results[hourly_results['gain/loss'] > 0].to_latex(f'{live_sim_results_dir}/successful_hourly_results.tex', header=True, index=False)
hourly_results.groupby('hour').mean()

round(hourly_results_buying, 2).to_latex(f'{live_sim_results_dir}/hourly_buying_results.tex', header=True, index=False)
round(hourly_results_buying[hourly_results_buying['gain/loss'] > 0], 2).to_latex(f'{live_sim_results_dir}/successful_hourly_buying_results.tex', header=True, index=False)
