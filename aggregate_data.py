import pandas as pd

def get_aggregated_mean(data, grouping, index_names=None):
    aggregated_mean = data.groupby(by=grouping).mean()
    if index_names:
        aggregated_mean.index = aggregated_mean.index.rename(index_names)
    aggregated_mean = aggregated_mean.reset_index()
    return aggregated_mean


def  merge_data(data, strategy_signal_data, merge_col, right_on, left_on, index):
    strategy_signal_data = data.merge(strategy_signal_data[merge_col], how='left', right_on=right_on, left_on=left_on)
    strategy_signal_data = strategy_signal_data.rename(columns={f"{merge_col}_x": merge_col, f"{merge_col}_y": f"{merge_col}_mean"})
    strategy_signal_data.index = strategy_signal_data[index]

    return strategy_signal_data

def offset_data_by_business_days(data_to_offset, n):
    return data_to_offset + pd.offsets.BusinessDay(n=n)
