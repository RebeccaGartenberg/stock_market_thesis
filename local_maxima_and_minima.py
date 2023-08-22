from plot_stock_data import plot
from dates import month_number_to_name
import pytz

def plot_local_maxima_and_minima(stock_symbol, sampled_data_list, start_date, time_axis, price_axis, dir_name):
    maxima, minima = findLocalMaximaMinima(len(price_axis), price_axis)
    maxima_quotes = [sampled_data_list[x] for x in maxima]
    minima_quotes = [sampled_data_list[x] for x in minima]

    est = pytz.timezone('US/Eastern')

    # For plotting the local maxima
    time_axis_maxima = []
    price_axis_maxima = []
    for quote in maxima_quotes:
        time_axis_maxima.append(quote.timestamp.astimezone(est))
        price_axis_maxima.append(quote.ask_price)

    # For plotting the local minima
    time_axis_minima = []
    price_axis_minima = []
    for quote in minima_quotes:
        time_axis_minima.append(quote.timestamp.astimezone(est))
        price_axis_minima.append(quote.ask_price)

    plot(x_axis=[time_axis_minima, time_axis_maxima],
        y_axis=[price_axis_minima, price_axis_maxima],
        x_axis_name='Timestamp (EST)',
        y_axis_name='Ask Price (USD)',
        legend_labels=['Local Maxima', 'Local Minima'],
        colors=['r', 'g'],
        marker='s',
        title=f"Stock: {stock_symbol} | {month_number_to_name(start_date.month)} {start_date.day}, {start_date.year}",
        show_plot=False,
        file_name=f"{dir_name}/{stock_symbol}_{start_date.month}_{start_date.day}_{start_date.year}_maxima_minima.png",
    )

def findLocalMaximaMinima(n, arr):
    '''
        Function to find all the local maxima and minima in the given array arr[]
        source: https://www.geeksforgeeks.org/find-indices-of-all-local-maxima-and-local-minima-in-an-array/
    '''
	# Empty lists to store points of local maxima and minima
    mx = []
    mn = []

    # Checking whether the first point is local maxima or minima or neither
    if(arr[0] > arr[1]):
    	mx.append(0)
    elif(arr[0] < arr[1]):
    	mn.append(0)

    # Iterating over all points to check local maxima and local minima
    for i in range(1, n-1):
    	# Condition for local minima
    	if(arr[i-1] > arr[i] < arr[i + 1]):
    		mn.append(i)
    	# Condition for local maxima
    	elif(arr[i-1] < arr[i] > arr[i + 1]):
    		mx.append(i)

    # Checking whether the last point is local maxima or minima or neither
    if(arr[-1] > arr[-2]):
    	mx.append(n-1)
    elif(arr[-1] < arr[-2]):
    	mn.append(n-1)
    return mx, mn
