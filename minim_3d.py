# Importing relevant packages
import numpy as np
from nll import NLL
        
np.random.seed(1234)  # Setting a random seed so that results can be compared across multiple runs

class MinimisationError(Exception):
    """Exception raised when the user attempts to calculate standard deviation without having minimised the function.

    Attributes:
        message: Explanation of the error.
    """
    def __init__(self, message = "Minimisation must have occurred before calculating the standard deviation!"):
        self.message = message
        print(self.message)  # Prints error message in console.

class Minimise3D():
    """Class which carries out 3-D minimisation, using either univariate or simultaneous minimisation methods.

    Given an input function (of which the most commonly used throughout this project is the NLL), and also optional
    minimisation parameters (in the case of the NLL), the parameter values which give the minimum value of the function
    can be found using the minimisation methods in this class.
    """
    def __init__(self, init_range_x, init_range_y, init_range_z, nll = True, nll_data = None, func = None, start_coord = None):
        """Initialisation of the Minimise2D class.

        Checks that the initialisation ranges (for the parameter in question) are of the correct format.
        Saves the input ranges and also any input data or input function given as private member variables.

        Args:
            init_range_x: Initial guess range for arbitrary x-parameter (mixing angle for the NLL case), in the form [lower_val, upper_val].
            init_range_y: Initial guess range for arbitrary y-parameter (squared mass diff. for the NLL case), in the form [lower_val, upper_val].
            init_range_z: Initial guess range for arbitrary z-parameter (cross-section prop. constant for the NLL case), in the form [lower_val, upper_val].
            nll: Flag to indicate that the function to be minimised is the NLL (i.e. parameters to be minimised for are θ_23 and squared mass diff.).
            nll_data: Data to be passed into NLL objects to be created.
            func: Function to be minimised (if not NLL).
            start_coord: Starting coordinate (default set to None). If not specified, the starting point of the various schemes is chosen at random.
        
        Raises:
            AttributeError: If the input ranges do not have 2 values each (lower, upper), or if both NLL and another function are
                            simultaneously chosen for minimisation.
            ValueError: If the input range is not in the form (lower, upper).
            ParameterError: If the function to be minimised is the NLL, but no parameters are specified.
        """
        # Checking for input errors
        if len(init_range_x) != 2 or len(init_range_y) != 2 or len(init_range_z) != 2:
            raise AttributeError("Input ranges must be a list or NumPy array of length 2!")
        if init_range_x[0] > init_range_x[1] or init_range_y[0] > init_range_y[1]:
            raise ValueError("Input ranges must be in the form [lower_value, upper_value].")
        if nll and func != None:
            raise AttributeError("Please only select one function to be minimised!")

        # Saving inputs as private member variables for later use
        self._init_range_x = init_range_x
        self._init_range_y = init_range_y 
        self._init_range_z = init_range_z
        self._data = nll_data  # Saves data needed for NLL calculation within the class
        self._nll = nll  # Boolean flag
        self._start_coord = start_coord
        if func is not None:
            self._func = func  # Saves function to be minimised within class
        

    def calc_nll(self, theta, mass, cross_sec):
        """Calculates the Negative Log Likelihood using the NLL class from nll.py.

        Creates an instance of the imported NLL class, in order to calculate the NLL value using the arguments given.
        
        Args:
            theta: Neutrino mixing angle.
            mass: Squared mass difference of neutrinos.
            cross_sec: Constant of proportionality/scaling factor of cross section increase with energy.
            
        Returns:
            nll: Value of NLL calculated.
        """
        # Defining default distance travelled by neutrino
        L = 295

        # Finding the NLL according to the input mixing angle or the squared mass difference. Uses data passed in during initialisation.
        nll_obj = NLL(energies = self._data[0], event_rates = self._data[1], obs_events = self._data[2], mix_ang = theta, distance = L, sq_mass_diff = mass, cross_sec = cross_sec)
        nll_obj.surv_prob()
        nll_obj.calc_lambda()
        nll = nll_obj.find_nll() 

        return nll  # Returns the NLL found

    def gen_init_points(self, param):
        """Calculates the initial values of a given parameter (x, y or z), and its corresponding function, to be used in the first parabolic minimisation.

        Finds 3 random values within the range specified for a given parameter (x, y or z), which can be used as the initial parameter values
        for the first iteration of the parabolic minimisation. 
        If the other parameter has not yet been minimised, uses the midpoint of its initial guess range to generate function values
        (using either the NLL calculation function calc_nll() or the function passed in during initialisation).
        These parameter and function values are saved as private member variables to be used by other methods.

        Args:
            param: Parameter for which the initial values are to be generated (choice between 'x', 'y', or 'z').

        Raises:
            ValueError: If the parameter argument entered is not in 'x', 'y', or 'z'.
        """
        # Checking for errors in the input
        if param not in ['x','y','z']:
            raise ValueError("Parameter specified must be either 'x', 'y' or 'z'!")

        # Choosing random points to start off the parabolic minimisation in the direction chosen
        if param == 'x':
            self._x = np.empty(3)
            if self._start_coord is not None:
                if self._iterations == 0:
                    # If starting coordinate specified
                    self._x[0] = self._start_coord[0]
                    self._ymin = self._start_coord[1]
                    self._zmin = self._start_coord[2]
                    for i in range(1,3):
                        # Choosing 2 more values close to the (x) starting coordinate
                        val = self._x[0] + (self._x[0] * 1e-4) * i
                        self._x[i] = val
                else:
                    for i in range(0, 3):
                        val = np.random.uniform(self._init_range_x[0], self._init_range_x[1]) 
                        self._x[i] = val
            else:
                # Choosing 3 random values within the (x) initialisation range
                for i in range(0, 3):
                    val = np.random.uniform(self._init_range_x[0], self._init_range_x[1]) 
                    self._x[i] = val
                if self._iterations == 0:
                    # If the first minimisation is in the x-direction, then we choose the y- and z- minima to be the midpoints
                    # of their given initialisation ranges (these will then later be overwritten during minimisation in those directions)
                    self._ymin = np.mean(self._init_range_y)  
                    self._zmin = np.mean(self._init_range_z)
        if param == 'y':
            self._y = np.empty(3)
            if self._start_coord is not None:
                if self._iterations == 0:
                    # If starting coordinate specified
                    self._xmin = self._start_coord[0]
                    self._y[0] = self._start_coord[1]
                    self._zmin = self._start_coord[2]
                    for i in range(1,3):
                        # Choosing 2 more values close to the (y) starting coordinate
                        val = self._y[0] + (self._y[0] * 1e-4) * i
                        self._y[i] = val
                else:
                    for i in range(0, 3):
                        val = np.random.uniform(self._init_range_y[0], self._init_range_y[1])
                        self._y[i] = val                   
            else:
                # Choosing 3 random values within the (y) initialisation range
                for i in range(0, 3):
                    val = np.random.uniform(self._init_range_y[0], self._init_range_y[1])
                    self._y[i] = val
                if self._iterations == 0:
                    # If the first minimisation is in the y-direction, then we choose the x- and z- minima to be the midpoints
                    # of their given initialisation ranges (these will then later be overwritten during minimisation in those directions)
                    self._xmin = np.mean(self._init_range_x)
                    self._zmin = np.mean(self._init_range_z)
        if param == 'z':
            self._z = np.empty(3)
            if self._start_coord is not None:
                if self._iterations == 0:
                    # If starting coordinate specified
                    self._xmin = self._start_coord[0]
                    self._ymin = self._start_coord[1]
                    self._z[0] = self._start_coord[2]
                    for i in range(1,3):
                        # Choosing 2 more values close to the (z) starting coordinate
                        val = self._z[0] + (self._z[0] * 8e-5) * i
                        self._z[i] = val
                else:
                    for i in range(0, 3):
                        val = np.random.uniform(self._init_range_z[0], self._init_range_z[1])
                        self._z[i] = val

            else:
                # Choosing 3 random values within the (z) initialisation range
                for i in range(0, 3):
                    val = np.random.uniform(self._init_range_z[0], self._init_range_z[1])
                    self._z[i] = val
                if self._iterations == 0:
                    # If the first minimisation is in the z-direction, then we choose the x- and y- minima to be the midpoints
                    # of their given initialisation ranges (these will then later be overwritten during minimisation in those directions)
                    self._xmin = np.mean(self._init_range_x)
                    self._ymin = np.mean(self._init_range_y)

        # Calculating function values for initial parabolic minimisation (f(x,y.z))
        self._f = np.empty(3)
        if param == 'x':
            self._start = self._x
            for ind, val in enumerate(self._x):
                if self._nll:
                    self._f[ind] = self.calc_nll(theta = val, mass = self._ymin, cross_sec = self._zmin)  # Generating NLL values for mixing angle minimisation
                else:
                    self._f[ind] = self._func(val, self._ymin, self._zmin)  # Generating function values (if the function to be minimised is not the NLL)
        if param == 'y':
            self._start = self._y
            for ind, val in enumerate(self._y):
                if self._nll:
                    self._f[ind] = self.calc_nll(theta = self._xmin, mass = val, cross_sec = self._zmin)  # Generating NLL values for square mass diff. minimisation
                else:
                    self._f[ind] = self._func(self._xmin, val, self._zmin)  # Generating function values (if the function to be minimised is not the NLL)
        if param == 'z':
            self._start = self._z
            for ind, val in enumerate(self._z):
                if self._nll:
                    self._f[ind] = self.calc_nll(theta = self._xmin, mass = self._ymin, cross_sec = val)  # Generating NLL values for square mass diff. minimisation
                else:
                    self._f[ind] = self._func(self._xmin, self._ymin, val)  # Generating function values (if the function to be minimised is not the NLL)       


    def univ_min(self, first = 'x'):
        """Univariate method for 3-D minimisation.

        Applies a parabolic minimisation algorithm in the first minimisation direction given until the minimum value is found.
        Then searches for the minimum in the second and third directions, using the minimum found for the previous directions as a function input.
        Minimises in alternating directions (x --> y --> z) until a convergence condition is satisfied, meaning that the overall minimum is found.

        Args:
            first: Direction to first search for the minimum in (Choice between 'x' and 'y').
        
        Raises:
            ValueError: If the first minimisation direction is not 'x', 'y', or 'z'.
        """
        # Checking for errors in the input
        if first not in ['x','y','z']:
            raise ValueError("Parameter specified must be 'x','y' or 'z'!")

        if first == 'x':
            xycounter = 0  # Counter needed to allow minimisation to occur in the other direction after every iteration
        if first == 'y':
            xycounter = 1
        if first == 'z':
            xycounter = 2

        self._overall_minimum_found = False  # Flag for the overall minimum being found (in all 3 directions)
        self._iterations = 0  # Total iteration counter
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        threshold = 1e-6  # Convergence condition threshold
        if self._start_coord is not None:
            self._mins_list.append(self._start_coord)
        # Initialising previous values of the minima in all directions (will be overwritten)
        prev_xmin = 1
        prev_ymin = 1
        prev_zmin = 1
        # Directional parabolic iteration and minimisation counters
        self._x_iters = 0
        self._y_iters = 0
        self._z_iters = 0
        self._min_iters_x = 0
        self._min_iters_y = 0
        self._min_iters_z = 0
        # Definining initial relative difference values (will be overwritten)
        self._rel_diff_x = 1
        self._rel_diff_y = 1
        self._rel_diff_z = 1

        # Outer while-loop: Iterates until the overall minimum is found
        while not self._overall_minimum_found:
            self._dir_iters = 0  # Iterations in given direction (counter resets after every directional minimum found)
            self._minimum_found = False  # Flag for directional minimum
            
            prev_min = 1  # Previous directional minimum (will be overwritten)
            # Inner while-loop: Iterates until a directional minimum is found
            while not self._minimum_found:  
                remainder = xycounter % 3  # Modulo operation to change the direction of minimisation with every increment of xycounter
                # If this is the first directional iteration, generate initial coordinate points and function values
                if self._dir_iters == 0:
                    if remainder == 0:
                        self.gen_init_points('x')  # Calculating three initial coordinate points for x-parameter to begin minimisation
                        coords = self._x
                        self._direction = 'x'  # Direction of minimisation
                    elif remainder == 1:
                        self.gen_init_points('y')  # Calculating three initial coordinate points for y-parameter to begin minimisation
                        coords = self._y
                        self._direction = 'y'
                    else:
                        self.gen_init_points('z')  # Calculating three initial coordinate points for z-parameter to begin minimisation
                        coords = self._z
                        self._direction = 'z'                       

                # Finding the minimum of the interpolating quadratic polynomial (P_2(x))
                numerator = ((coords[2] ** 2) - (coords[1] ** 2)) * self._f[0] + ((coords[0] ** 2) - (coords[2] ** 2)) * self._f[1] \
                            + ((coords[1] ** 2) - (coords[0] ** 2)) * self._f[2]
                denominator = (coords[2] - coords[1]) * self._f[0] + (coords[0] - coords[2]) * self._f[1] \
                                + (coords[1] - coords[0]) * self._f[2]
                minimum = 0.5 * numerator / denominator

                max_ind = np.argmax(self._f)  # Index of maximum function value
                coords[max_ind] = minimum  # Replace the coordinate value which gives the maximum function value, with the approximated minimum

                # Replacing the corresponding function value
                if self._direction == 'x':  # If currently minimising in x-direction
                    if self._nll:
                        self._f[max_ind] = self.calc_nll(minimum, self._ymin, self._zmin)  # Calls the calc_nll() function using previous directional minima
                    else:
                        self._f[max_ind] = self._func(minimum, self._ymin, self._zmin)  # Uses function passed into the minimisation object
                    self._mins_list.append([minimum, self._ymin, self._zmin])
                    self._x_iters += 1  # Incrementing x-direction iteration counter by 1
                elif self._direction == 'y':  # If currently minimising in y-direction
                    if self._nll:
                        self._f[max_ind] = self.calc_nll(self._xmin, minimum, self._zmin)  # Calls the calc_nll() function using previous directional minima
                    else:
                        self._f[max_ind] = self._func(self._xmin, minimum, self._zmin)
                    self._mins_list.append([self._xmin, minimum, self._zmin])
                    self._y_iters += 1  # Incrementing y-direction iteration counter by 1
                else:
                    if self._nll:
                        self._f[max_ind] = self.calc_nll(self._xmin, self._ymin, minimum)  # Calls the calc_nll() function using previous directional minima
                    else:
                        self._f[max_ind] = self._func(self._xmin, self._ymin, minimum)
                    self._mins_list.append([self._xmin, self._ymin, minimum])
                    self._z_iters += 1  # Incrementing z-direction iteration counter by 1
                
                if self._dir_iters == 0:  # No need to calculate relative difference for the first iteration
                    prev_min = minimum  # Set prev_min variable equal to current minimum for next iteration in this current direction
                else:
                    # Calculating relative difference between subsequent minima (in this current direction).
                    # If this difference is less than 0.001% of the previous minima, the flag is triggered and the while loop is exited.
                    rel_diff = abs(prev_min - minimum)/prev_min  
                    if rel_diff < threshold:
                        self._minimum_found = True  # Flag triggered
                        # Saves minimising parameter and minimum function value as private member variables
                        if self._direction == 'x':
                            self._xmin = minimum
                        elif self._direction == 'y':
                            self._ymin = minimum
                        else:
                            self._zmin = minimum

                        self._dir_min_func = self._f[max_ind]  # Directional minimum function value
                    else:
                        prev_min = minimum  # Set prev_min variable equal to current minimum for next iteration

                self._dir_iters += 1  # Increments current direction iteration counter by 1
                self._iterations += 1  # Increments total iteration counter by 1
                # End of inner while-loop

            xycounter += 1  # Counter incremented to change the direction of minimisation upon the next iteration of loop

            if self._direction == 'x':
                # If the inner loop has been iterating in the x-direction (i.e. x-minimum has just been found):
                if self._min_iters_x == 0:  # If first x-minimisation (i.e. no previous x-minimum available for comparison)
                    prev_xmin = self._xmin  # Sets previous x-minimum variable equal to the found minimum
                else: 
                    # Calculation of relative difference between successive x-direction minima
                    self._rel_diff_x = abs(prev_xmin - self._xmin)/prev_xmin  # Relative difference saved as private member variable
                    if self._rel_diff_x < threshold and self._rel_diff_y < threshold and self._rel_diff_z < threshold:
                        # Convergence condition: If x-, y-, and z- relative differences are below the threshold (less than 0.001% of previous minimum),
                        # then triggers the overall_minimum_found' flag and exits the loop after this iteration
                        self._overall_minimum_found = True
                        self._min = [self._xmin, prev_ymin, prev_zmin]  # Saves minimum (x,y,z) coordinate
                    else:
                        prev_xmin = self._xmin  # If convergence condition not met, sets previous x-minimum variable equal to the found minimum
                self._min_iters_x += 1  # Increments x-minimisation counter by 1
            elif self._direction == 'y':
                # If the inner loop has been iterating in the y-direction (i.e. y-minimum has just been found):
                if self._min_iters_y == 0:  # If first y-minimisation (i.e. no previous y-minimum available for comparison)
                    prev_ymin = self._ymin  # Sets previous y-minimum variable equal to the found minimum
                else: 
                    self._rel_diff_y = abs(prev_ymin - self._ymin)/prev_ymin
                    if self._rel_diff_x < threshold and self._rel_diff_y < threshold and self._rel_diff_z < threshold:
                        # Convergence condition
                        self._overall_minimum_found = True
                        self._min = [prev_xmin, self._ymin, prev_zmin]  # Saves minimum (x,y,z) coordinate
                    else:
                        prev_ymin = self._ymin  # If convergence condition not met, sets previous y-minimum variable equal to the found minimum
                self._min_iters_y += 1
            else:
                # If the inner loop has been iterating in the z-direction (i.e. z-minimum has just been found):
                if self._min_iters_z == 0:  # If first z-minimisation (i.e. no previous z-minimum available for comparison)
                    prev_zmin = self._zmin  # Sets previous z-minimum variable equal to the found minimum
                else: 
                    self._rel_diff_z = abs(prev_zmin - self._zmin)/prev_zmin
                    if self._rel_diff_x < threshold and self._rel_diff_y < threshold and self._rel_diff_z < threshold:
                        # Convergence condition
                        self._overall_minimum_found = True
                        self._min = [prev_xmin, prev_ymin, self._zmin]  # Saves minimum (x,y,z) coordinate
                    else:
                        prev_zmin = self._zmin  # If convergence condition not met, sets previous z-minimum variable equal to the found minimum
                self._min_iters_z += 1
            # End of outer while-loop
        self._nll_min = self.calc_nll(self._min[0], self._min[1], self._min[2])
        return self._min  # Returns coordinate containing minimising parameter
    
    def gen_start_pt(self):
        """Generates a starting point for iteration of the 3-D minimisation schemes (excluding univariate).

        Picks a random coordinate from the x-, y- and z- initialisation ranges used, and saves this coordinate as 
        a private member variable, in the form of a NumPy array.

        Returns:
            coord: Starting coordinate for the given minimisation scheme.
        """
        if self._start_coord is not None:
            coord = self._start_coord
        else:
            # Finds a random value within each of the initialisation ranges
            x_init = np.random.uniform(self._init_range_x[0], self._init_range_x[1])
            y_init = np.random.uniform(self._init_range_y[0], self._init_range_y[1])
            z_init = np.random.uniform(self._init_range_z[0], self._init_range_z[1])
            coord = np.array([x_init, y_init, z_init])  # Coordinate is expressed as a NumPy array of size 3
        return coord 

    def grad_min(self, alpha):
        """Gradient simultaneous minimisation method for 3 dimensions.

        Follows the steepest descent in gradient towards the minimum. This is done by calculating the gradient using a forward
        difference scheme, and taking a small step α in the direction opposite the gradient (as the gradient is perpendicular to the local contour line).
        However, α was scaled so that it has an equivalent relative magnitude in all coordinate directions, for optimal efficiency. 
        The coordinate is updated with each step taken, and iterations occur until the convergence condition is satisfied.

        Args:
            alpha: Size of step taken for each iteration of the descent.
        """
        self._iterations = 0  # Iteration counter
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_coord = self.gen_start_pt()  # Generating starting position
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        self._mins_list.append(self._prev_coord)
        h = 1e-6
        threshold = 1.5e-5  # Convergence condition threshold
        # As the parameters may have different orders of magnitude, we scale alpha to have the same relative size for all parameters
        # --> The scaling factors are found using the ratio between the means of the initialisation ranges
        # --> For NLL case, y (squared mass diff) is of a different order of magnitude and so a scaling factor is applied
        scaling = np.mean(self._init_range_x) / np.mean(self._init_range_y)
        alpha_x = alpha
        alpha_y = alpha_x / scaling
        alpha_z = alpha_x 
        alpha_vec = np.array([alpha_x, alpha_y, alpha_z])  # Alpha is expressed as a vector
        while not self._minimum_found:
            # Finding the gradient vector d, using a central differencing scheme
            d = np.empty(3)
            d[0] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) - \
                    self.calc_nll(self._prev_coord[0] - h, self._prev_coord[1], self._prev_coord[2])) / (2 * h)
            d[1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2]) - \
                    self.calc_nll(self._prev_coord[0], self._prev_coord[1] - h, self._prev_coord[2])) / (2 * h)
            d[2] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h) - \
                    self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] - h)) / (2 * h)
            new_coord = self._prev_coord - (alpha_vec * d)  # Calculation of new coordinate vector
            self._mins_list.append(new_coord)  # Saving new coordinate vector

            if self._iterations == 0:
                # No need to calculate relative difference for the first iteration
                self._prev_coord = new_coord  # Updating the coordinate for the next iteration
            else:
                # Calculation of relative difference in each direction between successive minima
                rel_diff_x = abs(self._prev_coord[0] - new_coord[0]) / self._prev_coord[0]
                rel_diff_y = abs(self._prev_coord[1] - new_coord[1]) / self._prev_coord[1]
                rel_diff_z = abs(self._prev_coord[2] - new_coord[2]) / self._prev_coord[2]
                if rel_diff_x < threshold and rel_diff_y < threshold and rel_diff_z < threshold:
                    # Convergence condition: If relative differences are below the threshold (less than 0.0001% of previous minimum),
                    # then triggers the 'minimum_found' flag and exits the loop after this iteration
                    self._minimum_found = True
                    self._min = new_coord  # Saving minimum
                    self._nll_min = self.calc_nll(new_coord[0], new_coord[1], new_coord[2])  # Calculating and saving the minimum NLL value at this minimum
                else:
                    self._prev_coord = new_coord  # Updating the coordinate for the next iteration if convergence condition is not met
            
            self._iterations += 1  # Incrementing iteration counter by 1
        
        return self._min  # Returning the coordinate vector that corresponds to the minimum function value

    def newton_min(self, alpha):
        """Newton simultaneous minimisation method for 3 dimensions.

        Takes the local curvature into account at each step for minimisation, by calculating the inverse Hessian and multiplying it by the 
        gradient vector to find the descent vector for each iteration.
        The coordinate is updated with each step taken, and iterations occur until the convergence condition is satisfied.

        Args:
            alpha: Size of step taken between each iteration.
        """
        self._iterations = 0  # Iteration counter
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_coord = self.gen_start_pt()  # Generating starting position
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        self._mins_list.append(self._prev_coord)
        threshold = 5e-7  # Convergence condition threshold
        h = 1e-6  # Step size for finite differencing (in this case the central-difference scheme)

        while not self._minimum_found:
            # Finding the gradient vector using central-differencing scheme
            grad = np.empty(3)
            grad[0] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) - \
                       self.calc_nll(self._prev_coord[0] - h, self._prev_coord[1], self._prev_coord[2])) / (2 * h)
            grad[1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2]) - \
                       self.calc_nll(self._prev_coord[0], self._prev_coord[1] - h, self._prev_coord[2])) / (2 * h)
            grad[2] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h) - \
                       self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] - h)) / (2 * h)

            hessian = np.empty((3,3))  # Initialising the Hessian matrix
            # Calculating each element of the Hessian matrix using forward-difference approximation
            hessian[0,0] = (self.calc_nll(self._prev_coord[0] + 2 * h, self._prev_coord[1], self._prev_coord[2]) - (2 * self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2])) + \
                            self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[0,1] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1] + h, self._prev_coord[2]) - self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2]) - \
                            self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) + self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[0,2] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2] + h) - self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h) - \
                            self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) + self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[1,0] = hessian[0,1]
            hessian[1,1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + 2 * h, self._prev_coord[2]) - (2 * self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2])) + \
                            self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[1,2] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2] + h) - self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h) - \
                            self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) + self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[2,1] = hessian[1,2]
            hessian[2,2] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + 2 * h) - (2 * self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h)) + \
                            self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)            
            
            # Calculating the next coordinate step
            new_coord = self._prev_coord - (alpha * np.matmul(grad, np.linalg.inv(hessian)))
            self._mins_list.append(new_coord)  # Saving new coordinate vector

            if self._iterations == 0: 
                # No need to calculate relative difference for the first iteration
                self._prev_coord = new_coord  # Updating the coordinate for the next iteration
            else:
                # Calculation of relative difference in each direction between successive minima
                rel_diff_x = abs(self._prev_coord[0] - new_coord[0]) / self._prev_coord[0]
                rel_diff_y = abs(self._prev_coord[1] - new_coord[1]) / self._prev_coord[1]
                rel_diff_z = abs(self._prev_coord[2] - new_coord[2]) / self._prev_coord[2]
                if rel_diff_x < threshold and rel_diff_y < threshold and rel_diff_z < threshold:
                    # Convergence condition: If relative differences are below the threshold (less than 0.0001% of previous minimum),
                    # then triggers the 'minimum_found' flag and exits the loop after this iteration
                    self._minimum_found = True
                    self._min = new_coord  # Saving minimum
                    self._nll_min = self.calc_nll(new_coord[0], new_coord[1], new_coord[2])  # Calculating and saving the minimum NLL value at this minimum
                else:
                    self._prev_coord = new_coord  # Updating the coordinate for the next iteration if convergence condition is not met

            self._iterations += 1  # Incrementing iteration counter by 1

        return self._min  # Returning the coordinate vector that corresponds to the minimum function value


    def quasi_newton_min(self, alpha):
        """Quasi-Newton simultaneous minimisation method for 3 dimensions.

        A less computationally intensive approximation of the Newton method, which uses the local gradient to approximate the inverse Hessian.
        However, the step size used, α, was scaled so that it has an equivalent relative magnitude in all coordinate directions, for optimal efficiency. 
        The coordinate is updated with each step taken, and iterations occur until the convergence condition is satisfied.

        Args:
            alpha: Size of step taken between each iteration.
        """
        self._iterations = 0  # Iteration counter
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_coord = self.gen_start_pt()  # Generating starting position
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        self._mins_list.append(self._prev_coord)
        threshold = 1e-7  # Convergence condition threshold
        # As the parameters may have different orders of magnitude, we scale alpha to have the same relative size for all parameters
        # --> The scaling factors are found using the ratio between the means of the initialisation ranges
        # --> For NLL case, y (squared mass diff) is of a different order of magnitude and so a scaling factor is applied
        scaling = np.mean(self._init_range_x) / np.mean(self._init_range_y)
        alpha_x = alpha
        alpha_y = alpha_x / scaling 
        alpha_z = alpha
        alpha_vec = np.array([alpha_x, alpha_y, alpha_z])  # Alpha is expressed as a vector
        G = np.identity(3)  # G (inverse Hessian approximation) is equal to the identity matrix for the first iteration.
        self._grad = np.empty(3)  # Initialising the gradient vector
        while not self._minimum_found:
            
            if self._iterations == 0:
                # Need to compute the gradient vector for the first iteration (grad is updated during the calculations of updating G for later iterations)
                # --> Central difference scheme used to compute gradient vector
                self._grad[0] = (self.calc_nll(self._prev_coord[0] + alpha_x, self._prev_coord[1], self._prev_coord[2]) - \
                                 self.calc_nll(self._prev_coord[0] - alpha_x, self._prev_coord[1], self._prev_coord[2])) / (2 * alpha_x)
                self._grad[1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + alpha_y, self._prev_coord[2]) - \
                                 self.calc_nll(self._prev_coord[0], self._prev_coord[1] - alpha_y, self._prev_coord[2])) / (2 * alpha_y)
                self._grad[2] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + alpha_z) - \
                                 self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] - alpha_z)) / (2 * alpha_z)

            new_coord = self._prev_coord - (alpha_vec * np.matmul(G, self._grad))  # Calculating the new coordinate for this step
            self._mins_list.append(new_coord)  # Saving new coordinate vector
            if self._iterations == 0:  # No need to check for convergence if first iteration
                self._prev_coord = new_coord  # Updating the coordinate for the next iteration
            else:
                # Calculation of relative difference in each direction between successive minima
                rel_diff_x = abs(self._prev_coord[0] - new_coord[0]) / self._prev_coord[0]
                rel_diff_y = abs(self._prev_coord[1] - new_coord[1]) / self._prev_coord[1]
                rel_diff_z = abs(self._prev_coord[2] - new_coord[2]) / self._prev_coord[2]
                if rel_diff_x < threshold and rel_diff_y < threshold and rel_diff_z < threshold:
                    # Convergence condition: If all relative differences are below the threshold (less than 0.00001% of previous minimum),
                    # then triggers the 'minimum_found' flag and exits the loop after this iteration
                    self._minimum_found = True
                    self._min = new_coord  # Saving minimum
                    self._nll_min = self.calc_nll(new_coord[0], new_coord[1], new_coord[2])  # Calculating and saving the minimum NLL value at this minimum
                else:
                    # If the initial convergence condition is not fulfilled
                    delta_n = new_coord - self._prev_coord  # Calculating the vector δ_n
                    new_grad = np.empty(3)
                    new_grad[0] = (self.calc_nll(new_coord[0] + alpha_x, new_coord[1], new_coord[2]) - \
                                   self.calc_nll(new_coord[0] - alpha_x, new_coord[1], new_coord[2])) / (2 * alpha_x)
                    new_grad[1] = (self.calc_nll(new_coord[0], new_coord[1] + alpha_y, new_coord[2]) - \
                                   self.calc_nll(new_coord[0], new_coord[1] - alpha_y, new_coord[2])) / (2 * alpha_y)
                    new_grad[2] = (self.calc_nll(new_coord[0], new_coord[1], new_coord[2] + alpha_z) - \
                                   self.calc_nll(new_coord[0], new_coord[1], new_coord[2] - alpha_z)) / (2 * alpha_y)
                    gamma_n = new_grad - self._grad  # Calculating the vector γ_n
                    gd_prod = np.dot(gamma_n, delta_n)  # Vector dot product of (γ_n, δ_n)
                    # Alternative convergence condition - if gamma_n * delta_n is equal to zero
                    # This also means that the step size is sufficiently small (G also cannot be updated due to division by zero)
                    if gd_prod == 0:
                        self._minimum_found = True
                        self._min = new_coord
                        self._nll_min = self.calc_nll(new_coord[0], new_coord[1], new_coord[2])
                    else:
                        # If there is no convergence, the vector G, ∇f, and the current coordinate are updated before the next iteration.
                        # Updating the inverse Hessian approximation matrix G using the DFP (Davidon-Fletcher-Powell) algorithm
                        outer_prod_d = np.outer(delta_n, delta_n)  # Outer product
                        next_G = G + (outer_prod_d/(gd_prod)) - \
                                  (np.matmul(G, (np.matmul(outer_prod_d, G))) / np.matmul(gamma_n, np.matmul(G, gamma_n)))
                        G = next_G
                        # Updating parameters for next iteration
                        self._prev_coord = new_coord  # Updating the coordinate for the next iteration if convergence condition is not met
                        self._grad = new_grad  # New gradient vector
                
            self._iterations += 1  # Incrementing iteration counter by 1

        return self._min  # Returning the coordinate vector that corresponds to the minimum function value  

    def LMA_min(self, alpha):
        """Levenberg–Marquardt Algorithm/Damped Least-squares simultaneous minimisation method for 3 dimensions.

        An algorithm which interpolates between the Gauss–Newton algorithm (GNA) and the gradient descent simultaneous minimisation method.
        At each iteration, calculates the minimisation error using the difference between the function value estimated from the Taylor
        series, and the true function value. This is expressed as a 'goodness of fit' parameter.
        If the error goes up upon the calculation of a new step, this means that we would want to follow the gradient of the function more
        --> the new step is rejected, and we scale alpha (the step size) up by 2.
        If the error goes down with the calculation of a new step, this means that we would want to accept the new step, however we would also
        want to reduce the influence of the gradient descent to prevent it becoming too large --> we scale alpha down by 2.
        The steps above are iterated until the convergence condition is reached.

        Args:
            alpha: Size of step taken between each iteration - this is scaled according to how good the fit is at each iteration.
                   Note that the step is denoted by lambda in the report.
        """
        self._iterations = 0  # Iteration counter
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_coord = self.gen_start_pt()  # Generating starting position
        # Initialising list of minima (for plotting purposes)
        self._mins_list = [] 
        self._mins_list.append(self._prev_coord)
        threshold = 1e-6  # Convergence condition threshold
        h = 1e-6  # Step size for finite differencing (in this case the central-difference scheme)
        while not self._minimum_found:
            # Finding the gradient vector using central-differencing scheme
            grad = np.empty(3)
            grad[0] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) - \
                       self.calc_nll(self._prev_coord[0] - h, self._prev_coord[1], self._prev_coord[2])) / (2 * h)
            grad[1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2]) - \
                       self.calc_nll(self._prev_coord[0], self._prev_coord[1] - h, self._prev_coord[2])) / (2 * h)
            grad[2] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h) - \
                       self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] - h)) / (2 * h)
            hessian = np.empty((3,3))  # Initialising the Hessian matrix
            # Calculating each element of the Hessian matrix using forward-difference approximation
            hessian[0,0] = (self.calc_nll(self._prev_coord[0] + 2 * h, self._prev_coord[1], self._prev_coord[2]) - (2 * self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2])) + \
                            self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[0,1] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1] + h, self._prev_coord[2]) - self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2]) - \
                            self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) + self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[0,2] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2] + h) - self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h) - \
                            self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) + self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[1,0] = hessian[0,1]
            hessian[1,1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + 2 * h, self._prev_coord[2]) - (2 * self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2])) + \
                            self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[1,2] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h, self._prev_coord[2] + h) - self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h) - \
                            self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1], self._prev_coord[2]) + self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)
            hessian[2,1] = hessian[1,2]
            hessian[2,2] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + 2 * h) - (2 * self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2] + h)) + \
                            self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2])) / (h**2)            

            # Calculating the proposed new coordinate for this step
            new_coord = self._prev_coord - np.matmul(np.linalg.inv(hessian + alpha * np.diag(np.diag(hessian))), grad)
            step = new_coord - self._prev_coord  # Step vector
            # Calculating the goodness of fit
            numerator = self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2]) - self.calc_nll(new_coord[0], new_coord[1], new_coord[2])
            # Second order Taylor Series estimate of function value
            taylor_est = self.calc_nll(new_coord[0], new_coord[1], new_coord[2]) + np.dot(grad, step) + \
                         0.5 * np.dot(step,np.matmul(hessian, step))
            denominator = self.calc_nll(self._prev_coord[0], self._prev_coord[1], self._prev_coord[2]) - taylor_est
            fit_goodness = numerator/denominator

            if self._iterations == 0: 
                # No need to calculate relative difference for the first iteration
                self._prev_coord = new_coord  # Updating the coordinate for the next iteration
            else:
                # If the goodness of fit is negative, we want to reject the step
                # --> New coordinate is not saved, and alpha is increased by a factor of 2
                if fit_goodness < 0:
                    alpha *= 2
                else:
                    # If the goodness of fit is positive, we want to accept the step
                    alpha /= 2               
                    # Calculation of relative difference in each direction between successive minima
                    rel_diff_x = abs(self._prev_coord[0] - new_coord[0]) / self._prev_coord[0]
                    rel_diff_y = abs(self._prev_coord[1] - new_coord[1]) / self._prev_coord[1]
                    rel_diff_z = abs(self._prev_coord[2] - new_coord[2]) / self._prev_coord[2]
                    if rel_diff_x < threshold and rel_diff_y < threshold and rel_diff_z < threshold:
                        # Convergence condition: If the differences are below the threshold,
                        # then triggers the 'minimum_found' flag and exits the loop after this iteration
                        self._minimum_found = True
                        self._min = new_coord  # Saving minimum
                        self._nll_min = self.calc_nll(new_coord[0], new_coord[1], new_coord[2])  # Calculating and saving the minimum NLL value at this minimum
                    else:
                        self._prev_coord = new_coord  # Updating the coordinate for the next iteration if convergence condition is not met
                        print(new_coord)

                    self._mins_list.append(new_coord)  # Appending new coordinate to list
            self._iterations += 1  # Incrementing iteration counter by 1

        return self._min  # Returning the coordinate vector that corresponds to the minimum function value

    def std_change(self, return_all = False):
        """Calculates the standard deviation of the minimising parameters using the change in the parabola.
        
        Each of the parameters are shifted incrementally in both directions, until the NLL has increased by an absolute value of 0.5.
        At this point, a shift of one standard deviation has occurred. To calculate the standard deviation, the shifts in both directions are averaged.
        There is also an option to measurements other than the standard deviation (namely θ+ and θ-, and their corresponding NLL values).

        Args:
            return_all: Returns all stats (standard deviation, θ+ and θ-, and their corresponding NLL values) in a list.
        
        Returns:
            self._std: Calculated standard deviation in each direction are returned in a list.
        
        Raises:
            MinimisationError: If the standard deviation method is called without minimisation previously occurring.
        """
        # Checking that minimisation has been carried out
        if not self._minimum_found:
            raise MinimisationError()
        
        self._std = []
        # Setting a limit for the NLL iterations - i.e. value of minimum NLL + 0.5 
        nll_lim = self._nll_min + 0.5
        steps_arr = [1e-5, 1e-7, 1e-5]  # List of increments used in the error calculation
        for ind, val in enumerate(steps_arr):
            self._var_plus = self._min[ind]
            self._plus_found = False
            while not self._plus_found:
                if ind == 0:
                    temp_nll_p = self.calc_nll(self._var_plus, self._min[1], self._min[2])  # Temporary calculated NLL value (in +ve direction)
                elif ind == 1:
                    temp_nll_p = self.calc_nll(self._min[0], self._var_plus, self._min[2])  # Temporary calculated NLL value (in +ve direction)
                else:
                    temp_nll_p = self.calc_nll(self._min[0], self._min[1], self._var_plus)  # Temporary calculated NLL value (in +ve direction)

                if temp_nll_p >= nll_lim:
                    self._plus_found = True  # If calculated NLL value is above the limit, triggers the flag
                else:
                    self._var_plus += val  # Increments theta_plus if NLL limit is not reached

            self._var_minus = self._min[ind]
            self._minus_found = False
            while not self._minus_found:
                if ind == 0:
                    temp_nll_m = self.calc_nll(self._var_minus, self._min[1], self._min[2])  # Temporary calculated NLL value (in -ve direction)
                elif ind == 1:
                    temp_nll_m = self.calc_nll(self._min[0], self._var_minus, self._min[2])  # Temporary calculated NLL value (in -ve direction)
                else:
                    temp_nll_m = self.calc_nll(self._min[0], self._min[1], self._var_minus)  # Temporary calculated NLL value (in -ve direction)   

                if temp_nll_m >= nll_lim:
                    self._minus_found = True  # If calculated NLL value is above the limit, triggers the flag
                else:
                    self._var_minus -= val  # Increments theta_plus if NLL limit is not reached

            # Finding the standard deviation by averaging the differences from the minimum
            std = ((self._var_plus - self._min[ind]) + (self._min[ind] - self._var_minus)) / 2 
            self._std.append(std)  # Variable consists of standard deviation only

        return self._std  # Returns standard deviations in all directions

    def std_gauss(self):
        """Calculates the standard deviation by approximating the NLL as a Gaussian distribution around the minimum.

        Finds the error in the (negative) log-likelihood for a single measurement, using the curvature (second derivative)
        of the function about the minimum.

        Returns:
            self._std_gauss: Standard deviation calculated using the Gaussian approximation.

        Raises:
            MinimisationError: If the standard deviation method is called without minimisation previously occurring.
        """
        # Checking that minimisation has been carried out
        if not self._minimum_found:
            raise MinimisationError()

        self._std_gauss = []  # Initialising list of standard deviations for all directions
        h = 1e-6  # Steps for central-difference scheme
        for i in range(3):
            # Calculating the second derivative of the NLL at the minimum for each direction, using forward-difference scheme
            if i == 0:
                second_derivative = (self.calc_nll(self._min[0] + 2 * h, self._min[1], self._min[2]) - (2 * self.calc_nll(self._min[0] + h, self._min[1], self._min[2])) + \
                           self.calc_nll(self._min[0], self._min[1], self._min[2])) / (h**2)
            elif i == 1:
                second_derivative =  (self.calc_nll(self._min[0], self._min[1] + 2 * h, self._min[2]) - (2 * self.calc_nll(self._min[0], self._min[1] + h, self._min[2])) + \
                            self.calc_nll(self._min[0], self._min[1], self._min[2])) / (h**2)
            else:
                second_derivative =  (self.calc_nll(self._min[0], self._min[1], self._min[2] + 2 * h) - (2 * self.calc_nll(self._min[0], self._min[1], self._min[2] + h)) + \
                            self.calc_nll(self._min[0], self._min[1], self._min[2])) / (h**2)

            std = 1/np.sqrt(second_derivative)
            self._std_gauss.append(std)
        
        return self._std_gauss

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
    def nll_min(self):
        return self._nll_min
    
    @property
    def min_iters_x(self):
        return self._min_iters_x
    
    @property
    def min_iters_y(self):
        return self._min_iters_y
    
    @property
    def min_iters_z(self):
        return self._min_iters_z

    @property
    def x_iters(self):
        return self._x_iters
    
    @property
    def y_iters(self):
        return self._y_iters

    @property
    def z_iters(self):
        return self._z_iters
    
    @property
    def mins_list(self):
        return self._mins_list
    
    @property
    def start(self):
        return self._start