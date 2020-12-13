# Importing relevant packages and modules
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os  # os is used in order to ensure that the correct file paths are accessed regardless of platform used

# Ensures that current working directory is the directory that contains the current module
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
cwd = os.getcwd()
plot_path = os.path.join(cwd, 'plots')  # Setting the path of the plots folder, that plots are to be saved to

class LengthError(Exception):
    """Exception raised when input arrays to the plotting function have unequal lengths.

    Attributes:
        message: Explanation of the error.
    """
    def __init__(self, message = "Input arrays must be equal in length!"):
        self.message = message
        print(self.message)  # Prints error message in console.

def plot_settings(clear = True, grid = True):
    """Defines the settings of the Matplotlib plot.

    Sets the size of the figure, and the fontsize of its elements. Options to clear the previous figure and to add a grid are available.

    Args:
        clear: Boolean argument to clear the previous figure (default set to True).
        grid: Boolean argument to include a grid in the figure (default set to True).
    """
    if clear:
        plt.clf()  # Clears any previous figures

    # Setting figure size
    figure = plt.gcf()
    figure.set_size_inches(18, 10)

    # Setting size of plot elements
    plt.rc('axes', labelsize = 22, titlesize = 24) 
    plt.rc('xtick', labelsize = 18)   
    plt.rc('ytick', labelsize = 18)    
    plt.rc('legend', fontsize = 20)
    plt.rc('axes', axisbelow = True) # Ensures that the grid is behind any graph elements
    if grid:
        plt.grid()  # Adds a grid to the plot

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
        title: Title of plot (default set to None).
        xlabel: x-axis label (default set to None).
        ylabel: y-axis label (default set to None).
        **bar_kwargs: Optional input arguments for the bar plot. If any of these are invalid, then an exception is raised within
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
    
    x_bins = np.linspace(0, 10, num = num_bins, endpoint = False)  # Generates list of x-values (in this case energies) at the start of each bin interval
    bin_size = x_bins[1] - x_bins[0]  # Calculating size of bins
    midpoints = [i + (bin_size/2) for i in x_bins]  # List of midpoints generated for the centre of each bar

    down_fact = len(input) / num_bins  # Calculating the downsampling factor
    if down_fact != 1:  
        down_arr = hist_downsample(input, down_fact)  # Downsampling the input data
    else:
        down_arr = input  # If the number of bins needed is equal to the length of the input data, no need to downsample

    plot_settings(grid = True)  # Defining plot settings
    plt.bar(midpoints, height = down_arr, width = bin_size, **bar_kwargs)  # Width of bar is set to the size of the histogram bin

    # Adds a title or axis labels if specified by the user
    if title != None:
        plt.title(title)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    
    # f-string allows save filepath to be set inside the plt.savefig() function
    plt.savefig(f'{os.path.join(plot_path,filename)}.pdf', dpi = 200)  # Saving the plot in the 'plots' folder (filepath set using the 'os' package)

def plot(x_input, y_input, filename, title = None, xlabel = None, ylabel = None, clear = True, legend = False, save = True, **plot_kwargs):
    """Creates and saves a plot of data from two 1-D arrays input by the user.

    Ensures that there are the same number of x-values as there are y-values before plotting data.
    Options to set the axes labels, as well as the title are available. Optional plot arguments are also supported.

    Args:
        x_input: Input list or NumPy array of x-coordinates.
        y_input: Input list or NumPy array of y-coordinates.
        filename: Filename of output plot (to be saved within the 'plots' folder).
        title: Title of plot (default set to None).
        xlabel: x-axis label (default set to None).
        ylabel: y-axis label (default set to None).
        clear: Clear previous figure(s) (default set to True).
        legend: Show legend on plot (default set to False).
        save: Saves figure according to filename argument (default set to True).
        **plot_kwargs: Optional input arguments for the plot. If any of these are invalid, then an exception is raised within
                      the Matplotlib package.

    Raises:
        LengthError: If the lengths of x_input and y_input are not equal.
    """
    # Checking that the x- and y- inputs are equal in length
    if len(x_input) != len(y_input):
        raise LengthError()

    plot_settings(clear = clear, grid = True)  # Defining plot settings
    plt.plot(x_input, y_input, **plot_kwargs)  # Plotting input values, with optional arguments

    # Adds a title or axis labels if specified by the user
    if title != None:
        plt.title(title)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    if legend:
        plt.legend()

    if save:    
        # f-string allows save filepath to be set inside the plt.savefig() function
        plt.savefig(f'{os.path.join(plot_path,filename)}.pdf', dpi = 200)  # Saving the plot in the 'plots' folder (filepath set using the 'os' package)

def surf_plot(x, y, z, filename, title = None, xlabel = None, ylabel = None, zlabel = None, elev = 0, azim = 0, **surf_kwargs):
    """Creates a surface plot based on three input arrays (x,y,z).

    Checks that the input arrays in all 3 cardinal directions are of the same length before plotting.
    Options to set the axes labels, as well as the title are available. Optional plot arguments are also supported.
    The 'camera angle' of the 3-dimensional surface plot can also be specified for the figure before saving.

    Args:
        x: Input list or NumPy array of x-coordinates.
        y: Input list or NumPy array of y-coordinates.
        z: Input list or NumPy array of z-coordinates.
        filename: Filename of output plot (to be saved within the 'plots' folder).
        title: Title of plot (default set to None).
        xlabel: x-axis label (default set to None).
        ylabel: y-axis label (default set to None). 
        zlabel: z-axis label (default set to None).
        elev: Elevation viewing angle (one of the 'camera angles') of the 3-D surface plot.
        azim: Azimuthal viewing angle (the other 'camera angle') of the 3-D surface plot.
        **surf_kwargs: Optional input arguments for the surface plot. If any of these are invalid, then an exception is raised within
                       the Matplotlib package.
    
    Raises:
        LengthError: If the lengths of the inputs (x, y and z) are not equal.
    """
    # Checking that the x- and y- and z- inputs are equal in length   
    if len(x) != len(y) != len(z):
        raise LengthError()

    fig = plt.figure()  # Creates blank figure
    ax = fig.gca(projection='3d')  # Creating 3-dimensional axes
    fig.set_size_inches(18, 10)  # Sets figure size

    # Plotting the surface - specifying the colormap, and setting the surface to opaque (with antialiased = False)
    ax.plot_trisurf(x, y, z, cmap = cm.coolwarm, linewidth=0, antialiased=False, **surf_kwargs) 

    # Setting plot parameters
    ax.set_title(title, fontsize = 24, pad = 15)
    ax.set_xlabel(xlabel, fontsize=18, labelpad = 15)
    ax.set_ylabel(ylabel, fontsize=18, labelpad = 15)
    ax.set_zlabel(zlabel, fontsize=18, labelpad = 15)
    ax.tick_params(axis='both', which='major', pad=10)
    ax.set_zlim(0, 1.0)  # z-axis limits set to [0,1] as the z-axis refers to probability in our case.

    ax.view_init(elev=elev, azim=azim)  # Sets 'camera angle' of surface plot, for saving
    # f-string allows save filepath to be set inside the plt.savefig() function
    plt.savefig(f'{os.path.join(plot_path,filename)}.pdf', dpi = 200)  # Saving the plot in the 'plots' folder
    
def spare():
    plt.show()
    plt.pause(40)
    print('ax.azim {}'.format(ax.azim))
    print('ax.elev {}'.format(ax.elev))
    