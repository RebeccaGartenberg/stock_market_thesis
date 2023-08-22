from plot_stock_data import plot
from dates import month_number_to_name
import pytz
import matplotlib.pyplot as plt
import numpy as np
import pdb
import matplotlib.dates as mdates
import pandas as pd

def plot_hourly_data(stock_symbol, year, time_axis, price_axis: list, dir_name):
    plot(x_axis=[time_axis],
        y_axis=price_axis,
        x_axis_name='Timestamp (EST)',
        y_axis_name='Ask Price (USD)',
        colors=[[0, 0.4470, 0.7410]],
        marker='o',
        linestyle='--',
        title=f"Stock: {stock_symbol} | {year} | Hourly Mean",
        show_plot=False,
        file_name=f"{dir_name}/{stock_symbol}_{year}_mean.png",
    )


def plot_hourly_mean_and_spread(stock_symbol, year, time_axis, price_axis: list, dir_name, show_plot: bool = False):
    x_axis_name='Timestamp (EST)'
    y_axis_name='Ask Price (USD)'
    title=f"Stock: {stock_symbol} | {year} | Mean and Spread"
    file_name=f"{dir_name}/{stock_symbol}_{year}_boxplot.png"

    time_axis_new = [time_axis[i].time() for i in range(0,len(time_axis))]
    ii = pd.DataFrame(time_axis_new)
    df = pd.DataFrame(data=price_axis).transpose()
    df.columns = ['09:00 AM', '10:00 AM', '11:00 AM', '12:00 PM', '1:00 PM', '2:00 PM', '3:00 PM']
    bp = df.boxplot(showmeans=True, grid=False, return_type='dict')

    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['medians'], color='blue')
    plt.setp(bp['means'], color='black') # color not changing
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red') # color  not changing
    plt.setp(bp['caps'], color='black')

    plt.xlabel(x_axis_name, fontsize=10, fontweight='bold')
    plt.ylabel(y_axis_name, fontsize=10, fontweight='bold')
    plt.title(title, fontweight='bold')
    est = pytz.timezone('US/Eastern')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], loc='upper right')

    if show_plot:
        plt.show()

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
