# Importing relevant package
import numpy as np
import random
from nll import NLL

class RangeError(Exception):
    """Exception raised when the entered initial guess range is invalid.

    Attributes:
        message: Explanation of the error.
    """
    def __init__(self, message = "Range is invalid. Please ensure it is within the range of the parameter values supplied."):
        self.message = message
        print(self.message)  # Prints error message in console.

class Minimise1D():
    """Class which carries out 1-D minimisation

    """
    def __init__(self, func_vals, param_arr, init_range):
        """Initialisation of the Minimise1D class.

        Input arguments are saved internally within the class, and used when the class methods are called.
        Ensures that function value and parameter arrays are of the same length.
        Checks that the guess parameter range is of the correct format, and within the range of the parameter array given.

        Args:
            func_vals: NumPy array containing values of the function for each value of the parameter (in our case the NLL).
            param_arr: Array containing values of the corresponding parameter for which the function is to be minimised.
            init_range: Initial guess parameter range which contains the function minimum in the form [lower_val, upper_val].
        
        Raises:
            AttributeError: If the input arrays are not of the same length.
            RangeError: If guess range is not within the range of the parameter array.
        """
        # Checking for input errors
        if len(func_vals) != len(param_arr):
            raise AttributeError("Input NLL and parameter arrays must be of equal length!")
        if not isinstance(func_vals, np.ndarray) and isinstance(param_arr, np.ndarray):
            raise TypeError("Please ensure that energies and event rates are in the form of NumPy arrays!")
        if len(init_range) != 2:
            raise AttributeError("Input range must be a list or NumPy array of length 2!")
        if init_range[0] < param_arr[0] or init_range[1] > param_arr[-1]:
            raise RangeError()
        if init_range[0] > init_range[1]:
            raise ValueError("Input range must be in the form [lower_value, upper_value].")

        # Saving the relevant data as private member variables for later use
        self._f = func_vals
        self._x = param_arr

        # Binary search of the parameter (x-) values to determine the initial range
        self._init_indices = []  # Initialising empty list
        self.search_closest(self._x, init_range)
    
    def para_min(self):
        """Parabolic minimiser method to find minimum of function.

        Choose 3 random points within the chosen range and fits a 2nd order Lagrange polynomial (P_2(x)) through these points.
        The minimum of this parabola is then found, and the point with the highest value out of the original 3 points and the
        approximated minimum is discarded. The process repeats with these 3 remaining points, until the difference between
        successive minima found is sufficiently small (0.1% of the previous).
        """
        # Choosing random existing points to start off the parabolic minimisation
        param_indices = np.empty(3)
        for i in range(0, 3):
            index = random.randint(self._init_indices[0], self._init_indices[2])
            param_indices[i] = index
        # Defining the exisiting points for the minimisation
        i0, i1, i2 = param_indices[0], param_indices[1], param_indices[2]
        x0, x1, x2 = self._x[i0], self._x[i1], self._x[i2]
        y0, y1, y2 = self._f[i0], self._f[i1], self._f[i2]
        
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_min = 0  # Previous value of the minimum (will be overwritten)

        while not self._minimum_found:
            # Finding the minimum of the interpolating quadratic polynomial (P_2(x))
            minimum = 0.5 * ( ((x2 ** 2) - (x1 ** 2)) * y0) + ((x0 ** 2))



    def search_closest(self, arr, target):
        """Searches for the closest values to the target in the array, and returns their indices.

        Uses a binary search (np.searchsorted()) to find the closest position(s) to the target number(s).
        Saves the closest indices to use as the starting range for the parabolic minimiser.
        This then allows us to use existing values to start the parabolic minimisation.

        Args:
            arr: Array on which the binary search is carried out.
            target: Target number(s) for which the closest value(s) in the array are to be found.
        """
        indices = np.searchsorted(arr, target)
        for i, val in enumerate(indices):
            prev_num = arr[val - 1]
            next_num = arr[val]
            if next_num - target[i] < target[i] - prev_num:
                self._init_indices.append(val)
                print(f"Difference from closest number in array({next_num}) = {next_num - target[i]}, index = {val}")
            else:
                self._init_indices.append(val - 1)
                print(f"Difference from closest number in array({prev_num}) = {target[i] - prev_num}, index = {val-1}")



num_thetas = 500
thetas = np.linspace(0, np.pi/2, num_thetas)  # Generating an array of mixing angles from [0,π/2]
nll_para = np.empty(num_thetas)  # Creating an empty NumPy array for the NLL values

# Finding the NLL for each mixing angle in theta_arr
for ind, val in enumerate(theta_arr):
    # Creating instance of NLL class, and finding lists of P and λ
    nll_theta = NLL(energies = en_array, event_rates = event_no, obs_events = exp_data, mix_ang = val, distance = L, sq_mass_diff = sq_mass_diff)
    nll_theta.surv_prob()
    nll_theta.calc_lambda()

    nll = nll_theta.find_nll()  # Calculating the NLL
    nll_para[ind] = nll  # Assigns the calculated NLL value with the corresponding index