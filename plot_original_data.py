from plot_stock_data import plot
from dates import month_number_to_name
import pytz

def plot_original_data_day(stock_symbol, start_date, time_axis, price_axis, dir_name, file_type='png'):
    plot(x_axis=[time_axis],
        y_axis=[price_axis],
        x_axis_name='Timestamp (EST)',
        y_axis_name='Ask Price (USD)',
        colors=[[0, 0.4470, 0.7410]],
        marker='o',
        linestyle='--',
        title=f"Stock: {stock_symbol} | {month_number_to_name(start_date.month)} {start_date.day}, {start_date.year}",
        show_plot=False,
        file_name=f"{dir_name}/{stock_symbol}_{start_date.month}_{start_date.day}_{start_date.year}.{file_type}",
    )

def plot_original_data_year(stock_symbol, year, time_axis, price_axis, dir_name, file_type='png'):
    plot(x_axis=[time_axis],
        y_axis=[price_axis],
        x_axis_name='Timestamp (EST)',
        y_axis_name='Ask Price (USD)',
        x_axis_format = "month",
        colors=[[0, 0.4470, 0.7410]],
        linestyle='-',
        title=f"Stock: {stock_symbol} | {year}",
        show_plot=False,
        file_name=f"{dir_name}/{stock_symbol}_{year}_original.{file_type}",
    )
