import matplotlib.pyplot as plt
import numpy as np
from plot_stock_data import plot
from dates import month_number_to_name

def plot_smoothed_data(stock_symbol, start_date, time_axis, price_axis, window, dir_name):
    average_price = generate_smoothed_data(time_axis, price_axis, window)
    plot(x_axis=[time_axis, time_axis],
        y_axis=[price_axis, average_price],
        x_axis_name='Timestamp (EST)',
        y_axis_name='Ask Price (USD)',
        legend_labels=['Original data', 'Moving average'],
        colors=['k', 'r'],
        marker='.',
        title=f"Stock: {stock_symbol} | {month_number_to_name(start_date.month)} {start_date.day}, {start_date.year} | Window size: {window}",
        show_plot=False,
        file_name=f"{dir_name}/{stock_symbol}_{start_date.month}_{start_date.day}_{start_date.year}_window_size_{window}.png",
    )

# https://learnpython.com/blog/average-in-matplotlib/
def generate_smoothed_data(x_axis, y_axis, window):
    '''
        simple moving average
    '''
    average_y = []
    for ind in range(len(y_axis) - window + 1):
        average_y.append(np.mean(y_axis[ind:ind+window]))
    for ind in range(window - 1):
        average_y.insert(0, np.nan)
    return average_y
