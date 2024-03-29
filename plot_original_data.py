from plot_stock_data import plot
from dates import month_number_to_name
import matplotlib.pyplot as plt

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
        y_axis_name='Close Price (USD)',
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
        title=f"Stock: {stock_symbol} | {year} | Stock Price with Simple Day Trading Indicators",
        x_axis_format="date",
        legend_labels=legend_labels,
        legend_fontsize=5,
        colors=['grey', 'red', 'blue'],
        # colors=['#bcbcbc', '#9fc5e8', '#0b5394', '#93c47d', '#38761d', '#ffd966', '#bf9000'],
        marker=[None, '.', '.'],
        linestyle=['-', 'None', 'None'],
        markersize=2,
        alpha=[None, 0.8, 0.8],
        file_name=f"{dir_name}/{stock_symbol}_{year}_baseline.{file_type}",
        show_plot=False
        )

def plot_original_data_trade_signals(stock_symbol, year, time_axis, price_axis, strategy, strategy_label, legend_labels, dir_name, file_type='png'):
    plot(x_axis=time_axis,
        y_axis=price_axis,
        x_axis_name='Timestamp (EST)',
        y_axis_name='Close Price (USD)',
        title=f"Stock: {stock_symbol} | {year} | Stock Price with {strategy_label} Trade Signals",
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

def plot_original_data_trade_signals_subplots(stock_symbol, year, time_axis, price_axis, strategy, y_axis_label, legend_labels, dir_name, file_type='png'):
    if y_axis_label in ['Mean Reversion', 'Hourly Mean Reversion']:
        y_axis_label_name = 'Z-score'
    else:
        y_axis_label_name = y_axis_label
    if strategy in ['hourly_slow_stoch', 'slow_stoch']:
        legend_loc = 'upper right'
    else:
        legend_loc='best'
    legend_fontsize=10
    file_name=f"{dir_name}/{stock_symbol}_{year}_{strategy}_trade_signals.{file_type}"
    show_plot=False
    colors=['grey', 'red', 'blue', 'green', 'black']
    marker=[None, None, None, None, None]
    linestyle=['-', '-', '-', '-', '-']
    alpha=[None, None, None, None, None]
    fig, axs = plt.subplots(2)
    axs[0].plot(price_axis[0], color=colors[0], marker=marker[0], linestyle=linestyle[0], label=legend_labels[0], alpha=alpha[0])
    for i in range(1, len(price_axis)):
        axs[1].plot(price_axis[i])
        axs[1].plot(price_axis[i], color=colors[i], marker=marker[i], linestyle=linestyle[i], linewidth=2, label=legend_labels[i], alpha=alpha[i])
    axs[0].set_ylabel('Close Price (USD)', fontsize=10, fontweight='bold')
    axs[1].set_ylabel(y_axis_label_name, fontsize=10, fontweight='bold')
    fig.text(0.5, 0.04, 'Timestamp (EST)', ha='center', va='center')
    axs[0].set_title(f"Stock: {stock_symbol} | {year} | Stock Price with {y_axis_label} Trade Signals", fontweight='bold')
    axs[0].legend(loc='best', fontsize=legend_fontsize)
    axs[1].legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
