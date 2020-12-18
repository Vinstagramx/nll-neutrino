# Importing relevant packages
import numpy as np
import re
import pandas as pd

class ShortcodeError(Exception):
    """Exception raised when the entered shortcode is not of the correct format.

    Attributes:
        message: Explanation of the error.
    """
    def __init__(self, message = "Shortcode is invalid. Please enter a valid Imperial shortcode."):
        self.message = message
        print(self.message)  # Prints error message in console.
        

def data_to_csv(shortcode):
    """Function which saves the experimental data from the High Energy Physics Group website as a .csv file.

    The user's shortcode is used to generate a personalised data file on the HEP Group's website. The shortcode is checked to ensure it is of
    the correct format using regex. Following this, the personalised data is downloaded and and saved as a two column .csv file. 

    Args:
        shortcode: Student's Imperial shortcode.
    
    Raises:
        ShortcodeError: If shortcode entered is invalid.
    """

    # Checking that shortcode is of correct format using regex (may be different for lecturers - CHECK)
    format_check = re.match("^[a-z]{2,3}[0-9]{0,5}(1[0-9]|20)$", shortcode)
    correct_format = bool(format_check)
    if not correct_format:
        raise ShortcodeError()
    
    data_url = f'http://www.hep.ph.ic.ac.uk/~ms2609/CompPhys/neutrino_data/{shortcode}.txt'
    fit_data = np.loadtxt(data_url, skiprows = 2, max_rows = 201)  # Reads the first half of the generated data (data to be fitted)
    unosc_flux = np.loadtxt(data_url, skiprows = 205)  # Reads the second half of the generated data (simulated event number prediction)

    # Saving data as .csv file - note that np.c_ saves the data in column format.
    np.savetxt(f"{shortcode}_data.csv", np.c_[fit_data, unosc_flux], delimiter=',', header = 'Data, Event Number', comments = '')

