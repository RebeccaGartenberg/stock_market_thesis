import pandas as pd
from dates import is_us_holiday

def get_rolling_hourly_mean(hour, aggregated_mean, *args):
    test = aggregated_mean[aggregated_mean['hour'] == hour].rolling(window=2).mean()
    test.rename(columns={'close': 'close_hourly_mean'}, inplace=True)

    return aggregated_mean.merge(test['close_hourly_mean'], how='left', left_on=[aggregated_mean.index.date, aggregated_mean.index.hour], right_on=[test.index.date, test.index.hour])


def get_aggregated_mean_hourly(data, n_days):
    aggregated_mean = data.resample('1H').mean()
    aggregated_mean['hour'] = aggregated_mean.index.hour
    aggregated_mean = aggregated_mean[aggregated_mean['close'].notna()]
    aggregated_mean['close_hourly_mean'] = float("nan")
    for hour in data.index.hour.unique():
        # Filter the aggregated_mean DataFrame for the specific hour
        hourly_aggregated_mean = aggregated_mean[aggregated_mean['hour'] == hour]

        # Calculate the rolling hourly mean for this hour and merge it back to the original DataFrame
        hourly_rolling_mean = hourly_aggregated_mean['close'].rolling(window=n_days).mean()
        aggregated_mean['close_hourly_mean'] = aggregated_mean['close_hourly_mean'].combine_first(hourly_rolling_mean)

    return aggregated_mean

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
