import yfinance as yf
import pdb
import pytz


def adjust_for_stock_split(data, stock_symbol, year):
    est = pytz.timezone('US/Eastern')

    split_info = yf.Ticker(stock_symbol).splits

    if isinstance(split_info, list) and split_info == []:
        return data

    split_info = split_info[split_info.index.year == year]
    split_info = split_info.reset_index()

    if not split_info.empty:
        for index, row in split_info.iterrows(): # make loop work but single works for now
            date = row['Date']
            split_ratio = row['Stock Splits']
            data.loc[data.index.tz_convert(est) >= date, 'close'] = data[data.index.tz_convert(est) >= date]['close'] * split_ratio

    return data
