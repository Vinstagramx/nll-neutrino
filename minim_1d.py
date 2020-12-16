# Importing relevant package
import numpy as np
from nll import NLL

class ParameterError(Exception):
    """Exception raised when the function to be minimised is the NLL, but no parameters are specified.

    Attributes:
        message: Explanation of the error.
    """
    def __init__(self, message = "Please specify either the 'theta' or the 'mass' parameters for NLL minimisation!"):
        self.message = message
        print(self.message)  # Prints error message in console.

class MinimisationError(Exception):
    """Exception raised when the user attempts to calculate standard deviation without having minimised the function.

    Attributes:
        message: Explanation of the error.
    """
    def __init__(self, message = "Parabolic minimisation must have occurred before calculating the standard deviation!"):
        self.message = message
        print(self.message)  # Prints error message in console.

class Minimise1D():
    """Class which carries out 1-D minimisation.

    Given an input function (of which the most commonly used throughout this project is the NLL), and also optional
    minimisation parameters (in the case of the NLL), the parameter value which gives the minimum value of the function
    can be found using the one-dimensional parabolic method.
    """

    def __init__(self, init_range, nll = True, nll_param = 'theta', nll_data = None, func = None):
        """Initialisation of the Minimise1D class.

        Calculates 3 random points within the initialisation range to begin the minimisation.
        Checks that the initialisation range (for the parameter in question) is of the correct format.

        Args:
            init_range: Initial guess parameter range which contains the function minimum in the form [lower_val, upper_val].
            nll: Flag to indicate that the function to be minimised is the NLL.
            nll_param: Parameter for which the minimum of the NLL is to be found for. 
            nll_data: Data to be passed into NLL objects to be created.
            func: Function to be minimised (if not NLL).
        
        Raises:
            AttributeError: If the input range does not have 2 values (lower, upper), or if both NLL and another function are
                            simultaneously chosen for minimisation.
            ValueError: If the input range is not in the form (lower, upper).
            ParameterError: If the function to be minimised is the NLL, but no parameters are specified.
        """
        # Checking for input errors
        if len(init_range) != 2:
            raise AttributeError("Input range must be a list or NumPy array of length 2!")
        if init_range[0] > init_range[1]:
            raise ValueError("Input range must be in the form [lower_value, upper_value].")
        if nll and func != None:
            raise AttributeError("Please only select one function to be minimised!")
        
        # Flags for NLL calculation in other class methods
        self._theta = False
        self._mass = False
        if nll_param == 'theta':
            self._theta = True
        if nll_param == 'mass':
            self._mass = True

        # Saving inputs as private member variables for later use
        self._init_range = init_range 
        self._data = nll_data
        self._nll = nll  # Boolean flag
        if func != None:
            self._func = func  # Saves function to be minimised within class

        if self._nll and not (self._theta or self._mass):
            raise ParameterError()  # Raises error if no parameters are specified for minimisation.
        
        # Calculating three initial coordinate points to begin minimisation
        self.gen_init_points()
        
    def para_min(self):
        """Parabolic minimiser method to find parameter value which gives function minimum.

        Uses 3 random points chosen the chosen range and fits a 2nd order Lagrange polynomial (P_2(x)) through these points.
        The minimum of this parabola is then found, and the point with the highest value out of the original 3 points and the
        approximated minimum is discarded. The process repeats with these 3 remaining points, until the difference between
        successive minima found is sufficiently small (0.1% of the previous).

        Returns:
            self._min: Value of the parameter which minimises the function.
        """
        self._minimum_found = False  # Flag for the minimum being found
        prev_min = 1  # Previous value of the minimum (will be overwritten)
        self._iterations = 0
        while not self._minimum_found:
            # Finding the minimum of the interpolating quadratic polynomial (P_2(x))
            numerator = ((self._x[2] ** 2) - (self._x[1] ** 2)) * self._y[0] + ((self._x[0] ** 2) - (self._x[2] ** 2)) * self._y[1] \
                        + ((self._x[1] ** 2) - (self._x[0] ** 2)) * self._y[2]
            denominator = (self._x[2] - self._x[1]) * self._y[0] + (self._x[0] - self._x[2]) * self._y[1] \
                            + (self._x[1] - self._x[0]) * self._y[2]
            minimum = 0.5 * numerator / denominator

            max_ind = np.argmax(self._y)  # Index of maximum y-value
            self._x[max_ind] = minimum  # Replace the x-value which gives the maximum y-value, with the approximated minimum
            # Replacing the corresponding y-value (i.e. f(x_min))
            if self._nll:
                self._y[max_ind] = self.calc_nll(minimum)  # Calls the calc_nll() function
            else:
                self._y[max_ind] = self._func(minimum)  # Uses function passed into the minimisation object
            
            if self._iterations == 0:  # No need to calculate relative difference for the first iteration
                prev_min = minimum  # Set prev_min variable equal to current minimum for next iteration
            else:
                # Calculating relative difference between subsequent minima.
                # If this difference is less than 0.1% of the previous minima, the flag is triggered and the while loop is exited.
                rel_diff = abs(prev_min - minimum)/prev_min  
                if rel_diff < 1e-3:
                    self._minimum_found = True  # Flag triggered
                    # Saves minimising parameter and minimum function value as private member variables
                    self._min = minimum
                    self._min_func = self._y[max_ind]
                else:
                    prev_min = minimum  # Set prev_min variable equal to current minimum for next iteration
            
            self._iterations += 1  # Increments iteration counter by 1
        
        return self._min  # Returns minimising parameter

    def calc_nll(self, val):
        """Calculates the Negative Log Likelihood using the NLL class from nll.py.

        Creates an instance of the imported NLL class, in order to calculate the NLL value for a given mixing angle
        or squared mass difference.

        Args:
            val: Value of mixing angle or squared mass difference for which the NLL is to be calculated for.

        Returns:
            nll: Value of NLL calculated.
        """
        # Defining default oscillation parameters
        L = 295
        if not self._theta:
            mix_ang = np.pi / 4  # Default mixing angle defined if it is not the parameter of interest
        if not self._mass:
            sq_mass_diff = 2.4e-3  # Default squared mass difference defined if it is not the parameter of interest

        # Finding the NLL when either the mixing angle or the squared mass difference is varied. Uses data passed in during initialisation.
        if self._theta:
            nll_obj = NLL(energies = self._data[0], event_rates = self._data[1], obs_events = self._data[2], mix_ang = val, distance = L, sq_mass_diff = sq_mass_diff)
            nll_obj.surv_prob()
            nll_obj.calc_lambda()
            nll = nll_obj.find_nll() 
        elif self._mass:
            nll_obj = NLL(energies = self._data[0], event_rates = self._data[1], obs_events = self._data[2], mix_ang = mix_ang, distance = L, sq_mass_diff = val)
            nll_obj.surv_prob()
            nll_obj.calc_lambda()
            nll = nll_obj.find_nll() 

        return nll  # Returns the NLL found

    def gen_init_points(self):
        """Calculates the initial x and y points to be used in the parabolic minimisation.

        Finds 3 random values within the range specified, which can be used as the initial parameter x-values
        for the first iteration of the parabolic minimisation. Also calculates the corresponding y-values by feeding the x-values into
        either the NLL calculation function calc_nll() or the function passed in during initialisation.
        These x- and y-values are saved as private member variables to be used by other methods.
        """
        # Choosing random x-points to start off the parabolic minimisation
        self._x = np.empty(3)
        for i in range(0, 3):
            val = np.random.uniform(self._init_range[0], self._init_range[1])
            self._x[i] = val

        # Calculating y-coordinates for initial parabolic minimisation
        self._y = np.empty(3)
        for ind, val in enumerate(self._x):
            if self._nll:
                self._y[ind] = self.calc_nll(val)
            else:
                self._y[ind] = self._func(val)
    
    def std_change(self, return_all = False):
        """Calculates the standard deviation of the minimising parameter (in this case the mixing angle) using the change in the parabola.
        
        The parameter (in this case the mixing angle) is shifted incrementally in both directions, until the NLL has increased by an absolute value of 0.5.
        At this point, a shift of one standard deviation has occurred. To calculate the standard deviation, the shifts in both directions are averaged.
        There is also an option to measurements other than the standard deviation (namely θ+ and θ-, and their corresponding NLL values).

        Args:
            return_all: Returns all stats (standard deviation, θ+ and θ-, and their corresponding NLL values) in a list.
        
        Returns:
            self._std_stats: Calculated standard deviation and related stats are returned in a list if return_all is set to True.
                             Only the standard deviation is returned if return_all is set to False.
        
        Raises:
            MinimisationError: If the standard deviation method is called without minimisation previously occurring.
        """
        # Checking that minimisation has been carried out
        if not self._minimum_found:
            raise MinimisationError()
        
        # Setting a limit for the NLL iterations - i.e. value of minimum NLL + 0.5 
        nll_lim = self._min_func + 0.5

        self._theta_plus = self._min  # θ+ initially set to minimising mixing angle value
        self._plus_found = False  # Boolean flag for θ+ being found
        while not self._plus_found:
            temp_nll_p = self.calc_nll(self._theta_plus)  # Temporary calculated NLL value (in +θ direction)
            if temp_nll_p >= nll_lim:
                self._plus_found = True  # If calculated NLL value is above the limit, triggers the flag
            else:
                self._theta_plus += 1e-5  # Increments theta_plus if NLL limit is not reached

        self._theta_minus = self._min  # θ- initially set to minimising mixing angle value
        self._minus_found = False  # Boolean flag for θ- being found
        while not self._minus_found:
            temp_nll_m = self.calc_nll(self._theta_minus)  # Temporary calculated NLL value (in -θ direction)
            if temp_nll_m >= nll_lim:
                self._minus_found = True  # If calculated NLL value is above the limit, triggers the flag
            else:
                self._theta_minus -= 1e-5  # Decrements theta_minus if NLL limit is not reached
        
        # Finding the standard deviation by averaging the differences of θ+ and θ- from the minimum
        std = ((self._theta_plus - self._min) + (self._min - self._theta_minus)) / 2  
        if return_all:
            self._std_stats = [std, self._theta_plus, self._theta_minus, temp_nll_p, temp_nll_m]  # Variable contains stats in list form
        else:
            self._std_stats = std  # Variable consists of standard deviation only

        return self._std_stats  # Returns stats variable

    def std_gauss(self):
        """Calculates the standard deviation by approximating the NLL as a Gaussian distribution around the minimum.

        Finds the error in the (negative) log-likelihood for a single measurement.
        The sample size is taken to be N = 200, as 200 distinct energy values are used to calculate the NLL.

        Returns:
            self._std_gauss: Standard deviation calculated using the Gaussian approximation.

        Raises:
            MinimisationError: If the standard deviation method is called without minimisation previously occurring.
        """
        # Checking that minimisation has been carried out
        if not self._minimum_found:
            raise MinimisationError()

        std = self._min / np.sqrt(200)
        self._std_gauss = std
        
        return self._std_gauss

    """
    Getters to access the private member variables outside the class.
    Necessary to display the number of iterations, and the minimum NLL value in the console.
    """    
    @property
    def iterations(self):
        return self._iterations
    
    @property
    def min_func(self):
        return self._min_func



