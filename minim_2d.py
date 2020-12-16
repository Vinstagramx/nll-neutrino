# Importing relevant packages
import numpy as np
from nll import NLL

class Minimise2D():
    """Class which carries out 2-D minimisation, using either the univariate or gradient methods.

    Given an input function (of which the most commonly used throughout this project is the NLL), and also optional
    minimisation parameters (in the case of the NLL), the parameter value which gives the minimum value of the function
    can be found using the one-dimensional parabolic method.
    """
    def __init__(self, init_range_x, init_range_y, nll = True, nll_data = None, func = None):
        """Initialisation of the Minimise1D class.

        Calculates 3 random points within the initialisation range to begin the minimisation.
        Checks that the initialisation range (for the parameter in question) is of the correct format.

        Args:
            init_range_x: Initial guess range for arbitrary x-parameter (mixing angle for the NLL case), in the form [lower_val, upper_val].
            init_range_y: Initial guess range for arbitrary y-parameter (squared mass diff. for the NLL case), in the form [lower_val, upper_val].
            nll: Flag to indicate that the function to be minimised is the NLL (i.e. parameters to be minimised for are θ_23 and squared mass diff.).
            nll_data: Data to be passed into NLL objects to be created.
            func: Function to be minimised (if not NLL).
        
        Raises:
            AttributeError: If the input range does not have 2 values (lower, upper), or if both NLL and another function are
                            simultaneously chosen for minimisation.
            ValueError: If the input range is not in the form (lower, upper).
            ParameterError: If the function to be minimised is the NLL, but no parameters are specified.
        """
        # Checking for input errors
        if len(init_range_x) != 2 or len(init_range_y) != 2:
            raise AttributeError("Input ranges must be a list or NumPy array of length 2!")
        if init_range_x[0] > init_range_x[1] or init_range_y[0] > init_range_y[1]:
            raise ValueError("Input ranges must be in the form [lower_value, upper_value].")
        if nll and func != None:
            raise AttributeError("Please only select one function to be minimised!")

        # Saving inputs as private member variables for later use
        self._init_range_x = init_range_x
        self._init_range_y = init_range_y 
        self._data = nll_data
        self._nll = nll  # Boolean flag
        if func != None:
            self._func = func  # Saves function to be minimised within class
        

    def calc_nll(self, theta, mass):
        """Calculates the Negative Log Likelihood using the NLL class from nll.py.

        Creates an instance of the imported NLL class, in order to calculate the NLL value for a given mixing angle
        or squared mass difference.
        
        Args:
            theta: Neutrino mixing angle.
            mass: Squared mass difference of neutrinos.

        Returns:
            nll: Value of NLL calculated.
        """
        # Defining default distance travelled by neutrino
        L = 295

        # Finding the NLL according to the input mixing angle or the squared mass difference. Uses data passed in during initialisation.
        nll_obj = NLL(energies = self._data[0], event_rates = self._data[1], obs_events = self._data[2], mix_ang = theta, distance = L, sq_mass_diff = mass)
        nll_obj.surv_prob()
        nll_obj.calc_lambda()
        nll = nll_obj.find_nll() 

        return nll  # Returns the NLL found

    def gen_init_points(self, param):
        """Calculates the initial values of a given parameter (x or y), and its corresponding function, to be used in the first parabolic minimisation.

        Finds 3 random values within the range specified for a given parameter (x or y), which can be used as the initial parameter values
        for the first iteration of the parabolic minimisation. 
        If the other parameter has not yet been minimised, uses the midpoint of its initial guess range to generate function values
        (using either the NLL calculation function calc_nll() or the function passed in during initialisation).
        These parameter and function values are saved as private member variables to be used by other methods.

        Args:
            param: Parameter for which the initial values are to be generated (choice between 'x', 'y').

        Raises:
            ValueError: If the parameter argument entered is neither 'x' nor 'y'.
        """
        # Checking for errors in the input
        if param not in ['x','y']:
            raise ValueError("Parameter specified must be either 'x' or 'y'!")

        # Choosing random points to start off the parabolic minimisation in the direction chosen
        if param == 'x':
            self._x = np.empty(3)
            # Choosing 3 random values within the (x) initialisation range
            for i in range(0, 3):
                val = np.random.uniform(self._init_range_x[0], self._init_range_x[1]) 
                self._x[i] = val
            if self._iterations == 0:
                # If the first minimisation is in the x-direction, then we choose the y-minimum to be the midpoint 
                # of the initialisation range given (this will then later be overwritten during y-direction minimisation)
                self._ymin = np.mean(self._init_range_y)  
        if param == 'y':
            self._y = np.empty(3)
            # Choosing 3 random values within the (y) initialisation range
            for i in range(0, 3):
                val = np.random.uniform(self._init_range_y[0], self._init_range_y[1])
                self._y[i] = val
            if self._iterations == 0:
                # If the first minimisation is in the y-direction, then we choose the x-minimum to be the midpoint 
                # of the initialisation range given (this will then later be overwritten during x-direction minimisation)
                self._xmin = np.mean(self._init_range_x)

        # Calculating function values for initial parabolic minimisation (f(x,y))
        self._f = np.empty(3)
        if param == 'x':
            for ind, val in enumerate(self._x):
                if self._nll:
                    self._f[ind] = self.calc_nll(theta = val, mass = self._ymin)  # Generating NLL values for mixing angle minimisation
                else:
                    self._f[ind] = self._func(val, self._ymin)
        if param == 'y':
            for ind, val in enumerate(self._y):
                if self._nll:
                    self._f[ind] = self.calc_nll(theta = self._xmin, mass = val)  # Generating NLL values for square mass diff. minimisation
                else:
                    self._f[ind] = self._func(self._xmin, val)


    def univ_min(self, first = 'x'):
        """Univariate method for 2-D minimisation.

        Applies a parabolic minimisation algorithm in the first minimisation direction given until the minimum value is found.
        Then searches for the minimum in the second direction, using the minimum found for the first direction as a function input.
        Minimises in alternating directions until a convergence condition is satisfied, meaning that the overall minimum is found.

        Args:
            first: Direction to first search for the minimum in (Choice between 'x' and 'y').
        
        Raises:
            ValueError: If the first minimisation direction is neither 'x' nor 'y'.
        """
        # Checking for errors in the input
        if first not in ['x','y']:
            raise ValueError("Parameter specified must be either 'x' or 'y'!")

        if first == 'x':
            xycounter = 0  # Counter needed to allow minimisation to occur in the other direction after every iteration
        if first == 'y':
            xycounter = 1

        self._overall_minimum_found = False  # Flag for the overall minimum being found (in both directions)
        self._iterations = 0  # Total iteration counter
        # Initialising previous values of the minima in both directions (will be overwritten)
        prev_xmin = 1
        prev_ymin = 1
        # x- and y- direction parabolic iteration and minimisation counters
        self._x_iters = 0
        self._y_iters = 0
        self._min_iters_x = 0
        self._min_iters_y = 0

        # Outer while-loop: Iterates until the overall minimum is found
        while not self._overall_minimum_found:
            self._dir_iters = 0  # Iterations in given direction (counter resets after every directional minimum found)
            self._minimum_found = False  # Flag for directional minimum
            
            prev_min = 1  # Previous directional minimum (will be overwritten)
            # Inner while-loop: Iterates until a directional minimum is found
            while not self._minimum_found:  
                remainder = xycounter % 2  # Modulo operation to change the direction of minimisation with every increment of xycounter
                # If this is the first directional iteration, generate initial coordinate points and function values
                if self._dir_iters == 0:
                    if remainder == 0:
                        self.gen_init_points('x')  # Calculating three initial coordinate points for x-parameter to begin minimisation
                        coords = self._x
                        self._direction = 'x'  # Direction of minimisation
                    else:
                        self.gen_init_points('y')  # Calculating three initial coordinate points for y-parameter to begin minimisation
                        coords = self._y
                        self._direction = 'y'

                # Finding the minimum of the interpolating quadratic polynomial (P_2(x))
                numerator = ((coords[2] ** 2) - (coords[1] ** 2)) * self._f[0] + ((coords[0] ** 2) - (coords[2] ** 2)) * self._f[1] \
                            + ((coords[1] ** 2) - (coords[0] ** 2)) * self._f[2]
                denominator = (coords[2] - coords[1]) * self._f[0] + (coords[0] - coords[2]) * self._f[1] \
                                + (coords[1] - coords[0]) * self._f[2]
                # print(numerator, denominator)
                minimum = 0.5 * numerator / denominator

                max_ind = np.argmax(self._f)  # Index of maximum function value
                coords[max_ind] = minimum  # Replace the coordinate value which gives the maximum function value, with the approximated minimum

                # Replacing the corresponding function value
                if self._direction == 'x':  # If currently minimising in x-direction
                    if self._nll:
                        self._f[max_ind] = self.calc_nll(minimum, self._ymin)  # Calls the calc_nll() function using previous y-minimum
                    else:
                        self._f[max_ind] = self._func(minimum, self._ymin)  # Uses function passed into the minimisation object
                    self._x_iters += 1  # Incrementing x-direction iteration counter by 1
                else:  # If currently minimising in y-direction
                    if self._nll:
                        self._f[max_ind] = self.calc_nll(self._xmin, minimum)  # Calls the calc_nll() function using previous x-minimum
                    else:
                        self._f[max_ind] = self._func(self._xmin, minimum)
                    self._y_iters += 1  # Incrementing x-direction iteration counter by 1
                
                if self._dir_iters == 0:  # No need to calculate relative difference for the first iteration
                    prev_min = minimum  # Set prev_min variable equal to current minimum for next iteration in this current direction
                else:
                    # Calculating relative difference between subsequent minima (in this current direction).
                    # If this difference is less than 0.001% of the previous minima, the flag is triggered and the while loop is exited.
                    rel_diff = abs(prev_min - minimum)/prev_min  
                    if rel_diff < 1e-5:
                        self._minimum_found = True  # Flag triggered
                        # Saves minimising parameter and minimum function value as private member variables
                        if self._direction == 'x':
                            self._xmin = minimum
                        else:
                            self._ymin = minimum

                        self._dir_min_func = self._f[max_ind]  # Directional minimum function value
                    else:
                        prev_min = minimum  # Set prev_min variable equal to current minimum for next iteration

                self._dir_iters += 1  # Increments current direction iteration counter by 1
                self._iterations += 1  # Increments total iteration counter by 1
                # End of inner while-loop

            xycounter += 1  # Counter incremented to change the direction of minimisation upon the next iteration of loop
            # If the inner loop has been iterating in the x-direction (i.e. x-minimum has just been found):
            if self._direction == 'x':
                if self._min_iters_x == 0:  # If first x-minimisation (i.e. no previous x-minimum available for comparison)
                    prev_xmin = self._xmin  # Sets previous x-minimum variable equal to the found minimum
                else: 
                    # Calculation of relative difference between successive x-direction minima
                    self._rel_diff_x = abs(prev_xmin - self._xmin)/prev_xmin  # Relative difference saved as private member variable
                    if self._rel_diff_x < 1e-5 and self._rel_diff_y < 1e-5:
                        # Convergence condition: If both x- and y- relative differences are below the threshold (less than 0.001% of previous minimum),
                        # then triggers the overall_minimum_found' flag and exits the loop after this iteration
                        self._overall_minimum_found = True
                        self._min = (self._xmin, prev_ymin)  # Saves minimum (x,y) coordinate as a tuple
                    else:
                        prev_xmin = self._xmin  # If convergence condition not met, sets previous x-minimum variable equal to the found minimum
                self._min_iters_x += 1  # Increments x-minimisation counter by 1

            # If the inner loop has been iterating in the y-direction (i.e. x-minimum has just been found):
            else:
                if self._min_iters_y == 0:  # If first y-minimisation (i.e. no previous y-minimum available for comparison)
                    prev_ymin = self._ymin  # Sets previous y-minimum variable equal to the found minimum
                else: 
                    self._rel_diff_y = abs(prev_ymin - self._ymin)/prev_ymin
                    if self._rel_diff_x < 1e-5 and self._rel_diff_y < 1e-5:
                        # Convergence condition: If both x- and y- relative differences are below the threshold (less than 0.001% of previous minimum),
                        # then triggers the overall_minimum_found' flag and exits the loop after this iteration
                        self._overall_minimum_found = True
                        self._min = (prev_xmin, self._ymin)  # Saves minimum (x,y) coordinate as a tuple
                    else:
                        prev_ymin = self._ymin
                self._min_iters_y += 1

        return self._min  # Returns coordinate tuple containing minimising parameter
    
    def gen_start_pt(self):
        """Generates a starting point for iteration of the 2-D minimisation schemes (excluding univariate).

        Picks a random coordinate from the x- and y- initialisation ranges used, and saves this coordinate as 
        a private member variable, in the form of a tuple.
        """
        x_init = np.random.uniform(self._init_range_x[0], self._init_range_x[1])
        y_init = np.random.uniform(self._init_range_y[0], self._init_range_y[1])
        self._coord = (x_init, y_init)

    def grad_min(self, alpha):
        """Gradient minimisation method for 2 dimensions.

        Follows the steepest descent in gradient towards the minimum. This is done by calculating the gradient using a forward
        difference scheme, and taking a small step α in the direction opposite the gradient (as the gradient is perpendicular to the local contour line).
        The coordinate is updated with each step taken, and iterations occur until the convergence condition is satisfied.
        """

        d = np.empty(2)  # Gradient vector
        d[0] = (self.calc_nll(self._coord[0] + alpha, self._coord[1]) - self.calc_nll(self._coord[0], self._coord[1]) / alpha)
        d[1] = (self.calc_nll(self._coord[0], self._coord[1] + alpha) - self.calc_nll(self._coord[0], self._coord[1]) / alpha)

    """
    Getters to access the private member variables outside the class.
    Necessary to display the number of iterations, and the minimum NLL value in the console.
    """    
    @property
    def iterations(self):
        return self._iterations
    
    @property
    def min(self):
        return self._min

    @property
    def dir_min_func(self):
        return self._dir_min_func
    
    @property
    def min_iters_x(self):
        return self._min_iters_x
    
    @property
    def min_iters_y(self):
        return self._min_iters_y

    @property
    def x_iters(self):
        return self._x_iters
    
    @property
    def y_iters(self):
        return self._y_iters
