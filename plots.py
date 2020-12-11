# Importing relevant packages
import matplotlib.pyplot as plt
import numpy as np
import os  # os is used in order to ensure that the correct file paths are accessed regardless of platform used

# Ensures that current working directory is the directory that contains the current module
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
cwd = os.getcwd()
plot_path = os.path.join(cwd, 'plots')  # Setting the path of the plots folder, that plots are to be saved to

def plot_settings(clear = True, grid = True):
    """Defines the settings of the Matplotlib plot.

    Clears the previous figure, and sets the size of the plot, and the fontsize of its elements.
    """
    if clear:
        plt.clf()  # Clears any previous figures
    figure = plt.gcf()  # Sets figure size
    figure.set_size_inches(18, 10)
    plt.rc('axes', labelsize = 22, titlesize = 24) 
    plt.rc('xtick', labelsize = 18)   
    plt.rc('ytick', labelsize = 18)    
    plt.rc('legend', fontsize = 20)
    plt.rc('axes', axisbelow = True) # Ensures that the grid is behind any graph elements
    if grid:
        plt.grid()


def hist_downsample(input, factor):
    """Downsamples the input bin frequency data, assuming equally spaced intervals.

    Adds consecutive elements of the bin frequency array, effectively reducing the number of datapoints by the downsampling factor.
    Returns an output array containing len(input)/factor elements.

    Args:
        input: Array of bin frequencies to be downsampled.
        factor: Integer downsampling factor, which is a factor of the input array length.
    
    Returns:
        downsampled: Downsampled array of bin frequencies.
    """
    factor = int(factor)  # Converts downsampling factor to type 'int', enabling easier indexing
    downsampled_len = round(len(input) / factor)  # Calculates integer length of downsampled data list
    downsampled = np.empty(downsampled_len)  # Create empty list
    for i in range(0, downsampled_len):
        # Finding the start and end indices of the original data which corresponds to the downsampled index
        start_index = i * factor
        end_index = start_index + factor
        # Summing over these indices to generate the downsampled data point
        sum = 0
        for j in input[start_index:end_index]:
            sum += j

        downsampled[i] = sum  # Assigns this new value to the downsampled list
    
    return np.array(downsampled)  # Returns the downsampled list as a NumPy array


def histogram(input, num_bins, filename, title = None, xlabel = None, ylabel = None, **bar_kwargs):
    """Creates and saves a histogram plot of the input (in this case our experimental data).

    Ensures that the number of bins inputted is an integer number.
    As data is already collated into bin format, a 'histogram' is generated in the form of a bar chart, over the range [0,10]
    where an array of bin intervals is created depending on the number of bins inputted.
    The input data is downsampled if needed. The plotted 'histogram' is saved in .pdf format (as it is a scalable vector graphic).

    Args:
        input: Input array of bin frequencies.
        num_bins: Number of input bins (must be a factor of the input length).
        filename: Filename of output plot (to be saved within the 'plots' folder).
        title: Title of plot.
        xlabel = X-axis label.
        ylabel = Y-axis label.
        bar_kwargs: Optional input arguments for the bar plot. If any of these are invalid, then an exception is raised within
                    the Matplotlib package.

    Raises:
        TypeError: If a non-integer number of bins is entered.
        ValueError: If the number of bins desired by the user is not a factor of the original data length
                    (i.e. resampling cannot occur).

    """ 
    # Checking for errors in the input
    if not isinstance(num_bins, int):
        raise TypeError('Please enter an integer number of bins.')
    if 200 % num_bins != 0:
        raise ValueError('The original number of intervals must be divisible by the number of bins inputted.')
    
    x_bins = np.linspace(0, 10, num = num_bins)
    bin_size = x_bins[1] - x_bins[0]
    midpoints = [i + (bin_size/2) for i in x_bins]  # List of midpoints generated for the centre of each bar

    down_fact = len(input) / num_bins  # Calculating the downsampling factor
    if down_fact != 1:  # If the number of bins needed is equal to the length of the input data, no need to downsample
        down_arr = hist_downsample(input, down_fact)  # Downsampling the input data
    else:
        down_arr = input

    plot_settings(grid = True)
    plt.bar(midpoints, height = down_arr, width = bin_size, **bar_kwargs)  # Width of bar is set to the size of the histogram bin

    # If user specifies axis labels
    if title != None:
        plt.title(title)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt,ylabel(ylabel)
    
    # f-string allows save filepath to be set inside the plt.savefig() function
    plt.savefig(f'{os.path.join(plot_path,filename)}.pdf', dpi = 200)  # Saving the plot in the 'plots' folder

def plot(x_input, y_input, filename, title = None, **plot_kwargs):
    """check x and y same langth
    """
    plot_settings(grid = True)
    plt.plot(x_input, y_input, **plot_kwargs)
    if title != None:
        plt.title(title)
        
    # f-string allows save filepath to be set inside the plt.savefig() function
    plt.savefig(f'{os.path.join(plot_path,filename)}.pdf', dpi = 200)  # Saving the plot in the 'plots' folder