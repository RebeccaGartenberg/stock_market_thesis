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

def plot_original_data_year_with_trade_markers(stock_symbol, year, time_axis, price_axis, legend_labels, dir_name, file_type='png'):
    plot(x_axis=time_axis,
        y_axis=price_axis,
        x_axis_name='Timestamp (EST)',
        y_axis_name='Close Price (USD)',
        title=f"Stock: {stock_symbol} | {year} | Stock Price with Trade Indicators",
        x_axis_format="date",
        legend_labels=legend_labels,
        legend_fontsize=5,
        colors=['grey', 'red', 'indianred', 'blue', 'cornflowerblue', 'darkorange', 'moccasin', 'green', 'yellowgreen', 'purple', 'violet',\
            'yellow', 'gold', 'pink', 'hotpink', 'chocolate', 'saddlebrown', 'turquoise', 'lightseagreen'],
        # colors=['#bcbcbc', '#9fc5e8', '#0b5394', '#93c47d', '#38761d', '#ffd966', '#bf9000'],
        marker=[None, '^', 'v', '^', 'v', '^', 'v', '^', 'v', '^', 'v', '^', 'v', '^', 'v', '^', 'v', '^', 'v'],
        linestyle=['-', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None'],
        markersize=2,
        alpha=[None, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        # alpha=[None, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        file_name=f"{dir_name}/{stock_symbol}_{year}_trade_markers.{file_type}",
        show_plot=False
        )

def plot_original_data_trade_signals(stock_symbol, year, time_axis, price_axis, strategy, legend_labels, dir_name, file_type='png'):
    plot(x_axis=time_axis,
        y_axis=price_axis,
        x_axis_name='Timestamp (EST)',
        y_axis_name='Close Price (USD)',
        title=f"Stock: {stock_symbol} | {year} | Stock Price with Trade Indicators",
        x_axis_format="date",
        legend_labels=legend_labels,
        legend_fontsize=10,
        file_name=f"{dir_name}/{stock_symbol}_{year}_{strategy}_trade_signals.{file_type}",
        show_plot=False,
        colors=['grey', 'red', 'blue', 'green'],
        marker=[None, None, None, None],
        linestyle=['-', '-', '-', '-'],
        alpha=[None, None, None, None],

        )
