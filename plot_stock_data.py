import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import pandas as pd

def plot(x_axis: list,
        y_axis: list[list],
        x_axis_name: str,
        y_axis_name: str,
        title: str,
        y_lim: list=[],
        x_axis_format: str = "time",
        error_bars=None,
        legend_labels: list[str] = None,
        colors: list[str] = None,
        marker: list[str] = None,
        linestyle: list[str] = None,
        markersize: int = None,
        alpha: list[int] = None,
        file_name: str = None,
        show_plot: bool = True):
    # format x axis to dispay time of day in est standard time
    est = pytz.timezone('US/Eastern')
    if x_axis_format == "time":
        xformatter = mdates.DateFormatter('%I:%M %p', tz=est)
    elif x_axis_format == "month":
        xformatter = mdates.DateFormatter('%b')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, bymonthday=1, interval=1, tz=None))
    elif x_axis_format == 'date':
        xformatter = mdates.DateFormatter('%m/%d/%Y')

    plt.xticks(rotation=45)
    plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
    x_axis_plot = x_axis[0]

    # superimpose multiple plots
    if isinstance(x_axis, pd.Series):
        for plot_num in range(0,len(y_axis)):
            plt.plot(y_axis[plot_num], color=colors[plot_num], marker=marker[plot_num], linestyle=linestyle[plot_num], label=legend_labels[plot_num], markersize=markersize, alpha=alpha[plot_num])
    else:
        for plot_num in range(0,len(y_axis)):
            if isinstance(x_axis, list) and len(x_axis) > 1:
                x_axis_plot = x_axis[plot_num]
            line, = plt.plot(x_axis_plot, y_axis[plot_num], color=colors[plot_num], marker=marker, linestyle=linestyle)
            if legend_labels:
                line.set_label(legend_labels[plot_num])
    if error_bars is not None:
        plt.errorbar(x_axis[0], y_axis[0], error_bars, linestyle='None', marker='o')
        # plt.fill_between(x_axis[0], (y_axis[0]-error_bars).tolist(), (y_axis[0]+error_bars).tolist())
    plt.xlabel(x_axis_name, fontsize=10, fontweight='bold')
    plt.ylabel(y_axis_name, fontsize=10, fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.legend(loc='best')
    if len(y_lim) == 2:
        plt.ylim(y_lim[0], y_lim[1])
    if show_plot:
        plt.show()

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
