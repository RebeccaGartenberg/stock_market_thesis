from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
import pytz
from plot_original_data import plot_original_data_year
from plot_hourly_data import plot_hourly_data, plot_hourly_mean_and_spread
import yaml

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
stock_symbol_list = params.get("stock_symbols")
plot_directory = params.get("plot_directory")
data_path = params.get("data_path")
year = params.get("year")

est = pytz.timezone('US/Eastern')
time_axis = [datetime(year, 1, 1, 9,0,0).astimezone(est), datetime(year, 1, 1, 10,0,0).astimezone(est), datetime(year, 1, 1, 11,0,0).astimezone(est),
datetime(year, 1, 1, 12,0,0).astimezone(est), datetime(year, 1, 1, 13,0,0).astimezone(est), datetime(year, 1, 1, 14,0,0).astimezone(est),
datetime(year, 1, 1, 15,0,0).astimezone(est), datetime(year, 1, 1, 16,0,0).astimezone(est)] # #time_axis.append(quote.timestamp.astimezone(est))

start_of_trading_day = datetime(2022, 1, 1, 9, 30, 0).time() # 9:30am EST
end_of_trading_day = datetime(2022, 1, 1, 16, 0, 0).time() # 4:00pm EST
start_date = datetime(year, 1, 1)
end_date = datetime(year, 12, 31).replace(hour=17, minute=0, second=0, microsecond=0, tzinfo = est)

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

for stock_symbol in stock_symbol_list:
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

    hourly_means = df.groupby(df.tz_convert(est).index.hour).mean()

    value_list = [[],[],[],[],[],[],[],[]]
    for count, value in enumerate(df.groupby(df.tz_convert(est).index.hour)):
        value_list[count] = value[1]['close']

    # average change in price from beginning to end of hour
    hour_begin_prices = df.groupby([df.tz_convert(est).index.date, df.tz_convert(est).index.hour])['close'].apply(lambda x: x.iloc[0])
    hour_end_prices = df.groupby([df.tz_convert(est).index.date, df.tz_convert(est).index.hour])['close'].apply(lambda x: x.iloc[-1])
    hourly_percent_change = ((hour_end_prices - hour_begin_prices)/hour_begin_prices).reset_index()
    hourly_average_change = hourly_percent_change.groupby(hourly_percent_change['timestamp']).mean()

    # Plot raw data
    plot_original_data_year(stock_symbol, year, df.index, df['close'], plot_directory, 'svg')

    # Plot hourly mean data with standard deviation
    plot_hourly_data(stock_symbol, year, time_axis[0:len(hourly_means)], hourly_means['close'], plot_directory, 'svg', y_lim=[hourly_means['close'].min()-0.5, hourly_means['close'].max()+0.5])

    # Plot hourly mean with spread
    plot_hourly_mean_and_spread(stock_symbol, year, value_list.index, value_list, plot_directory, file_type='svg', y_lim=[df['close'].min(), df['close'].max()])

    # Plot hourly percent change
    plot_hourly_data(stock_symbol, year, time_axis[0:len(hourly_means)], hourly_average_change, plot_directory, 'svg', f'{stock_symbol}_{year}_pct_change', 'Percent Change', y_lim=[hourly_average_change.min()['close']-0.0005, hourly_average_change.max()['close']+0.0005])
