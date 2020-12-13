# Importing relevant package
import numpy as np
import random
from nll import NLL

class ParameterError(Exception):
    """Exception raised when the function to be minimised is the NLL, but no parameters are specified.

    Attributes:
        message: Explanation of the error.
    """
    def __init__(self, message = "Please specify either the 'theta' or the 'mass' parameters for NLL minimisation!"):
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
        self._prev_min = 1  # Previous value of the minimum (will be overwritten)
        self._iterations = 0
        while not self._minimum_found:
            # Finding the minimum of the interpolating quadratic polynomial (P_2(x))
            numerator = ((self._x[2] ** 2) - (self._x[1] ** 2)) * self._y[0] + ((self._x[0] ** 2) - (self._x[2] ** 2)) * self._y[1] \
                        + ((self._x[1] ** 2) - (self._x[0] ** 2)) * self._y[2]
            denominator = (self._x[2] - self._x[1]) * self._y[0] + (self._x[0] - self._x[2]) * self._y[1] \
                            + (self._x[1] - self._x[0]) * self._y[2]
            minimum = 0.5 * numerator / denominator

            max_ind = np.argmax(self._y)  # Index of maximum y-value
            self._x[max_ind] = minimum  # Replace the x-value which gives the maximum y-value with the approximated minimum
            # Replacing the corresponding y-value (i.e. f(x_min))
            if self._nll:
                self._y[max_ind] = self.calc_nll(minimum)  # Calls the calc_nll() function
            else:
                self._y[max_ind] = self._func(minimum)  # Uses function passed into the minimisation object
            
            # Calculating relative difference between subsequent minima.
            # If this difference is less than 0.1% of the previous minima, the flag is triggered and the while loop is exited.
            rel_diff = abs(self._prev_min - minimum)/self._prev_min  
            if rel_diff < 1e-3 and self._iterations > 0:
                self._minimum_found = True  # Flag triggered
                # Saves minimising parameter and minimum function value as private member variables
                self._min = minimum
                self._min_func = self._y[max_ind]
            else:
                self._prev_min = minimum  # Set prev_min variable equal to current minimum for next iteration
            
            self._iterations += 1  # Increments iteration counter by 1
        
        return self._min  # Returns minimising parameter

    def calc_nll(self, val):
        """Calculates the Negative Log Likelihood using the NLL class from nll.py.

        Creates an instance of the imported NLL class, in order to calculate the NLL value for a given mixing angle
        or squared mass difference.

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

        Finds 3 random values within the range specified during initialisation, so they can be used as the initial parameter x-values
        for the first iteration of the parabolic minimisation. Also calculates the corresponding y-values by feeding the x-values into
        either the NLL calculation function calc_nll() or the function passed in during initialisation.
        These x- and y-values are saved as private member variables to be used by other methods.
        """
        # Choosing random existing points to start off the parabolic minimisation
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


