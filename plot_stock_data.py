import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz

def plot(x_axis: list,
        y_axis: list[list],
        x_axis_name: str,
        y_axis_name: str,
        title: str,
        x_axis_format: str = "time",
        legend_labels: list[str] = None,
        colors: list[str] = None,
        marker: str = None,
        linestyle: str = None,
        file_name: str = None,
        show_plot: bool = True):
    # format x axis to dispay time of day in est standard time
    est = pytz.timezone('US/Eastern')
    if x_axis_format == "time":
        xformatter = mdates.DateFormatter('%I:%M %p', tz=est)
    elif x_axis_format == "month":
        xformatter = mdates.DateFormatter('%b')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, bymonthday=1, interval=1, tz=None))

    plt.xticks(rotation=45)
    plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
    x_axis_plot = x_axis[0]
    # superimpose multiple plots
    for plot_num in range(0,len(y_axis)):
        if isinstance(x_axis, list) and len(x_axis) > 1:
            x_axis_plot = x_axis[plot_num]
        line, = plt.plot(x_axis_plot, y_axis[plot_num], color=colors[plot_num], marker=marker, linestyle=linestyle)
        if legend_labels:
            line.set_label(legend_labels[plot_num])
    plt.xlabel(x_axis_name, fontsize=10, fontweight='bold')
    plt.ylabel(y_axis_name, fontsize=10, fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.legend(loc='best')
    if show_plot:
        plt.show()

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
