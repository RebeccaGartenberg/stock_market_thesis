from plot_stock_data import plot
from dates import month_number_to_name
import pytz
import matplotlib.pyplot as plt
import numpy as np
import pdb
import matplotlib.dates as mdates
import pandas as pd

def plot_hourly_data(stock_symbol, year, time_axis, price_axis: list, dir_name, file_type='png', file_name=None, plot_type='Mean', error_bars=None, y_lim=[]):
    plot(x_axis=[time_axis],
        y_axis=[price_axis],
        x_axis_name='Timestamp (EST)',
        y_axis_name='Ask Price (USD)',
        error_bars=error_bars,
        y_lim=y_lim,
        colors=[[0, 0.4470, 0.7410]],
        marker='o',
        linestyle='--',
        title=f"Stock: {stock_symbol} | {year} | Hourly {plot_type}",
        show_plot=False,
        file_name= f"{dir_name}/{file_name}.{file_type}" if file_name is not None else f"{dir_name}/{stock_symbol}_{year}_mean.{file_type}",
    )

def plot_hourly_mean_and_spread(stock_symbol, year, time_axis, price_axis: list, dir_name, show_plot: bool = False, file_type='png', y_lim=[]):
    x_axis_name='Timestamp (EST)'
    y_axis_name='Ask Price (USD)'
    title=f"Stock: {stock_symbol} | {year} | Mean and Spread"
    file_name=f"{dir_name}/{stock_symbol}_{year}_boxplot.{file_type}"

    df = pd.DataFrame(data=price_axis).transpose()
    columns = ['09:00 AM', '10:00 AM', '11:00 AM', '12:00 PM', '1:00 PM', '2:00 PM', '3:00 PM', '4:00 PM']
    df.columns = columns[0:len(price_axis)]
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

def plot_hourly_number_of_trades(stock_symbol, hourly_total_trades, methods, year, dir_name, file_type='png', show_plot: bool = False):
    plt.xlabel('Hour (EST)')
    plt.ylabel('Number of Trades')
    hour_labels = [f"{hour % 12} AM" if hour < 12 else f"{hour % 12 if hour > 12 else hour} PM" for hour in range(9, 18)]
    plt.xticks(range(9, 18), hour_labels)
    plt.title(f'Stock: {stock_symbol} | {year} | Total trades per hour')
    plt.legend(loc='best')
    file_name=f"{dir_name}/{stock_symbol}_{year}_hourly_trades.{file_type}"
    colors = ['red', 'indianred', 'blue', 'cornflowerblue', 'darkorange', 'moccasin', 'green', 'yellowgreen', 'purple', 'violet',\
        'yellow', 'gold', 'pink', 'hotpink', 'chocolate', 'saddlebrown']

    if isinstance(hourly_total_trades, list):
        for i, ht in enumerate(hourly_total_trades):
            plt.bar(ht[0].index+0.05*(2*i), ht[0].values, width=0.05, label=f'{methods[i]} Buy', align='center', alpha=0.7, color=colors[2*i])
            plt.bar(ht[1].index+0.05*(2*i+1), ht[1].values, width=0.05, label=f'{methods[i]} Sell', align='center', alpha=0.7, color=colors[(2*i)+1])
    else:
        plt.bar(hourly_total_trades[0].index, hourly_total_trades[0].values, width=0.4, label='Buys', align='center', alpha=0.7)
        plt.bar(hourly_total_trades[1].index+0.4, hourly_total_trades[1].values, width=0.4, label='Sells', align='center', alpha=0.7)

    plt.legend(loc='best', fontsize=5)

    # Save or show figure
    if show_plot:
        plt.show()

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

def plot_hourly_profits(stock_symbol, hourly_profits, methods, year, dir_name, file_type='png', show_plot: bool = False):
    plt.xlabel('Hour (EST)')
    plt.ylabel('Profits per hour (USD)')
    hour_labels = [f"{hour % 12} AM" if hour < 12 else f"{hour % 12 if hour > 12 else hour} PM" for hour in range(9, 18)]
    plt.xticks(range(9, 18), hour_labels)# Save or show figure
    plt.title(f'Stock: {stock_symbol} | {year} | Total profits per hour')
    file_name=f"{dir_name}/{stock_symbol}_{year}_hourly_profits.{file_type}"
    colors = ['grey','red', 'cornflowerblue', 'darkorange', 'green', 'violet', 'gold', 'pink', 'chocolate']

    if isinstance(hourly_profits, list):
        for i, hp in enumerate(hourly_profits):
            plt.bar(hp.index+0.1*i, hp.values, width=0.1, label=f'{methods[i]}', align='center', alpha=0.7, color=colors[i])
    else:
        plt.bar(hourly_profits.index, hourly_profits.values)
    plt.legend(loc='best', fontsize=5)

    # Save or show figure
    if show_plot:
        plt.show()

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

def plot_hourly_returns(stock_symbol, hourly_profits, methods, year, dir_name, file_type='png', show_plot: bool = False):
    plt.xlabel('Hour (EST)')
    plt.ylabel('Returns per hour (%)')
    hour_labels = [f"{hour % 12} AM" if hour < 12 else f"{hour % 12 if hour > 12 else hour} PM" for hour in range(9, 18)]
    plt.xticks(range(9, 18), hour_labels)# Save or show figure
    plt.title(f'Stock: {stock_symbol} | {year} | Total profits per hour')
    file_name=f"{dir_name}/{stock_symbol}_{year}_hourly_profits.{file_type}"
    colors = ['grey','red', 'cornflowerblue', 'darkorange', 'green', 'violet', 'gold', 'pink', 'chocolate']

    if isinstance(hourly_profits, list):
        for i, hp in enumerate(hourly_profits):
            plt.bar(hp.index+0.1*i, hp.values, width=0.1, label=f'{methods[i]}', align='center', alpha=0.7, color=colors[i])
    else:
        plt.bar(hourly_profits.index, hourly_profits.values)
    plt.legend(loc='best', fontsize=5)

    # Save or show figure
    if show_plot:
        plt.show()

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
