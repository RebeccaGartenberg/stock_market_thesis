import pandas as pd
from dates import is_us_holiday

def get_aggregated_mean_hourly(data, n_days, operation='mean', col='close'):
    if operation == 'mean':
        aggregated_data = data.resample('1H').mean()
        col_name = 'close_hourly_mean'
    elif operation == 'min':
        aggregated_data = data.resample('1H').min()
        col_name = 'lowest_low'
    elif operation == 'max':
        aggregated_data = data.resample('1H').max()
        col_name = 'highest_high'
    elif operation == 'count':
        aggregated_data = data.resample('1H').count()
        col_name = 'count'

    if col == 'gain':
        col_name = 'avg_gain'
    elif col == 'loss':
        col_name = 'avg_loss'

    aggregated_data['hour'] = aggregated_data.index.hour
    aggregated_data = aggregated_data[aggregated_data[col].notna()]
    aggregated_data[f'{col_name}'] = float("nan")
    for hour in data.index.hour.unique():
        # Filter the aggregated_mean DataFrame for the specific hour
        hourly_aggregated_data = aggregated_data[aggregated_data['hour'] == hour]

        # Calculate the rolling hourly mean for this hour and merge it back to the original DataFrame
        if operation == 'mean':
            hourly_rolling_data = hourly_aggregated_data[col].rolling(window=n_days).mean()
        elif operation == 'min':
            hourly_rolling_data = hourly_aggregated_data[col].rolling(window=n_days).min()
        elif operation == 'max':
            hourly_rolling_data = hourly_aggregated_data[col].rolling(window=n_days).max()
        elif operation == 'count':
            hourly_rolling_data = hourly_aggregated_data[col].rolling(window=n_days).sum()

        aggregated_data[f'{col_name}'] = aggregated_data[f'{col_name}'].combine_first(hourly_rolling_data)

    return aggregated_data

def get_aggregated_mean(data, grouping, n_day_average=1, index_names=None):
    aggregated_mean = data.groupby(by=grouping).mean()
    if index_names:
        aggregated_mean.index = aggregated_mean.index.rename(index_names)
    aggregated_mean = aggregated_mean.rolling(window=n_day_average).mean()
    aggregated_mean = aggregated_mean.reset_index()

    return aggregated_mean


def merge_data(data, strategy_signal, merge_col, right_on, left_on, rename_cols=False, index=None):
    strategy_signal_data = data.merge(strategy_signal[merge_col], how='left', right_on=right_on, left_on=left_on)
    if rename_cols:
        strategy_signal_data = strategy_signal_data.rename(columns={f"{merge_col}_x": merge_col, f"{merge_col}_y": f"{merge_col}_mean"})
    if index:
        strategy_signal_data.index = strategy_signal_data[index]

    return strategy_signal_data

def offset_data_by_business_days(data_to_offset, n):
    offset_date = data_to_offset + pd.offsets.BusinessDay(n=n)
    while offset_date[offset_date.apply(is_us_holiday)].any():
        offset_date[offset_date.apply(is_us_holiday)] = offset_date[offset_date.apply(is_us_holiday)] + pd.offsets.BusinessDay(n=1)
    return offset_date
