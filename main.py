from download_data import data_to_csv  # Importing method to download data using shortcode
import plots  # Plotting module
# Importing relevant packages
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Defining colour list (to use in plots)
colour_list = list(mcolors.TABLEAU_COLORS.values())  # Using the Tableau colours from matplotlib.colors

"""
Section 3.1 - Generating Personalised Data and Preliminary Analysis
--> Are we supposed to generate histograms for the expected event rate???
"""
shortcode = 'wyw18'
# data_to_csv(shortcode)  # Generate and download data file from shortcode

# Loading the data that was saved as a .csv file
data = np.loadtxt(f'{shortcode}_data.csv', delimiter = ',', skiprows = 1).T  # Transpose to load the data saved in column form
exp_data = data[0]
event_no = data[1]

# Plotting histograms of the data over the range 0-10 GHz, with varying numbers of bins (and thus varying intervals)
bins_list = [200, 100, 50, 25]
for i, val in enumerate(bins_list):
    plots.histogram(exp_data, val, f"hist_{val}", f"Plot of Experimental Data ({val} bins)", xlabel = 'Energy (GeV)', ylabel = 'Frequency' color = colour_list[i])

"""
Section 3.2 - Calculating Oscillation Probability and Investigating Oscillation Parameters
"""



