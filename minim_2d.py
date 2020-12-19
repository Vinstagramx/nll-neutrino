# Importing relevant packages
import numpy as np
from nll import NLL

class Minimise2D():
    """Class which carries out 2-D minimisation, using either univariate or simultaneous minimisation methods.

    Given an input function (of which the most commonly used throughout this project is the NLL), and also optional
    minimisation parameters (in the case of the NLL), the parameter values which give the minimum value of the function
    can be found using the minimisation methods in this class.
    """
    def __init__(self, init_range_x, init_range_y, nll = True, nll_data = None, func = None, start_coord = None):
        """Initialisation of the Minimise2D class.

        Checks that the initialisation ranges (for the parameter in question) are of the correct format.
        Saves the input ranges and also any input data or input function given as private member variables.

        Args:
            init_range_x: Initial guess range for arbitrary x-parameter (mixing angle for the NLL case), in the form [lower_val, upper_val].
            init_range_y: Initial guess range for arbitrary y-parameter (squared mass diff. for the NLL case), in the form [lower_val, upper_val].
            nll: Flag to indicate that the function to be minimised is the NLL (i.e. parameters to be minimised for are θ_23 and squared mass diff.).
            nll_data: Data to be passed into NLL objects to be created.
            func: Function to be minimised (if not NLL).
            start_coord: Starting coordinate (default set to None). If not specified, the starting point of the various schemes is chosen at random.
        
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
        self._data = nll_data  # Saves data needed for NLL calculation within the class
        self._nll = nll  # Boolean flag
        self._start_coord = start_coord
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
            if self._start_coord is not None:
                if self._iterations == 0:
                    # If starting coordinate specified
                    self._x[0] = self._start_coord[0]
                    self._ymin = self._start_coord[1]
                    for i in range(1,3):
                        # Choosing 2 more random values within the (x) initialisation range
                        val = np.random.uniform(self._init_range_x[0], self._init_range_x[1]) 
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
                    # If the first minimisation is in the x-direction, then we choose the y-minimum to be the midpoint 
                    # of the initialisation range given (this will then later be overwritten during y-direction minimisation)
                    self._ymin = np.mean(self._init_range_y)  
        if param == 'y':
            self._y = np.empty(3)
            if self._start_coord is not None:
                if self._iterations == 0:
                    # If starting coordinate specified
                    self._xmin = self._start_coord[0]
                    self._y[0] = self._start_coord[1]
                    for i in range(1,3):
                        # Choosing 2 more random values within the (y) initialisation range
                        val = np.random.uniform(self._init_range_y[0], self._init_range_y[1]) 
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
                    self._f[ind] = self._func(val, self._ymin)  # Generating function values (if the function to be minimised is not the NLL)
        if param == 'y':
            for ind, val in enumerate(self._y):
                if self._nll:
                    self._f[ind] = self.calc_nll(theta = self._xmin, mass = val)  # Generating NLL values for square mass diff. minimisation
                else:
                    self._f[ind] = self._func(self._xmin, val)  # Generating function values (if the function to be minimised is not the NLL)


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
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        threshold = 1e-5
        if self._start_coord is not None:
            self._mins_list.append(self._start_coord)
        # Initialising previous values of the minima in both directions (will be overwritten)
        prev_xmin = 1
        prev_ymin = 1
        # x- and y- direction parabolic iteration and minimisation counters
        self._x_iters = 0
        self._y_iters = 0
        self._min_iters_x = 0
        self._min_iters_y = 0
        # Definining initial relative difference values (will be overwritten)
        self._rel_diff_x = 1
        self._rel_diff_y = 1

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
                minimum = 0.5 * numerator / denominator

                max_ind = np.argmax(self._f)  # Index of maximum function value
                coords[max_ind] = minimum  # Replace the coordinate value which gives the maximum function value, with the approximated minimum

                # Replacing the corresponding function value
                if self._direction == 'x':  # If currently minimising in x-direction
                    if self._nll:
                        self._f[max_ind] = self.calc_nll(minimum, self._ymin)  # Calls the calc_nll() function using previous y-minimum
                    else:
                        self._f[max_ind] = self._func(minimum, self._ymin)  # Uses function passed into the minimisation object
                    self._mins_list.append([minimum, self._ymin])
                    self._x_iters += 1  # Incrementing x-direction iteration counter by 1
                else:  # If currently minimising in y-direction
                    if self._nll:
                        self._f[max_ind] = self.calc_nll(self._xmin, minimum)  # Calls the calc_nll() function using previous x-minimum
                    else:
                        self._f[max_ind] = self._func(self._xmin, minimum)
                    self._mins_list.append([self._xmin, minimum])
                    self._y_iters += 1  # Incrementing y-direction iteration counter by 1
                
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
                    if self._rel_diff_x < threshold and self._rel_diff_y < threshold:
                        # Convergence condition: If both x- and y- relative differences are below the threshold (less than 0.001% of previous minimum),
                        # then triggers the overall_minimum_found' flag and exits the loop after this iteration
                        self._overall_minimum_found = True
                        self._min = [self._xmin, prev_ymin]  # Saves minimum (x,y) coordinate 
                    else:
                        prev_xmin = self._xmin  # If convergence condition not met, sets previous x-minimum variable equal to the found minimum
                self._min_iters_x += 1  # Increments x-minimisation counter by 1

            # If the inner loop has been iterating in the y-direction (i.e. x-minimum has just been found):
            else:
                if self._min_iters_y == 0:  # If first y-minimisation (i.e. no previous y-minimum available for comparison)
                    prev_ymin = self._ymin  # Sets previous y-minimum variable equal to the found minimum
                else: 
                    self._rel_diff_y = abs(prev_ymin - self._ymin)/prev_ymin
                    if self._rel_diff_x < threshold and self._rel_diff_y < threshold:
                        # Convergence condition: If both x- and y- relative differences are below the threshold (less than 0.001% of previous minimum),
                        # then triggers the 'overall_minimum_found' flag and exits the loop after this iteration
                        self._overall_minimum_found = True
                        self._min = [prev_xmin, self._ymin]  # Saves minimum (x,y) coordinate 
                    else:
                        prev_ymin = self._ymin  # If convergence condition not met, sets previous y-minimum variable equal to the found minimum
                self._min_iters_y += 1
            # End of outer while-loop
        self._mins_list.append(self._min)  # Appending final minimum to list of minima
        return self._min  # Returns coordinate tuple containing minimising parameter
    
    def gen_start_pt(self):
        """Generates a starting point for iteration of the 2-D minimisation schemes (excluding univariate).

        Picks a random coordinate from the x- and y- initialisation ranges used, and saves this coordinate as 
        a private member variable, in the form of a NumPy array.

        Returns:
            coord: Starting coordinate for the given minimisation scheme.
        """
        if self._start_coord is not None:
            # If starting coordinate is specified
            coord = self._start_coord
        else:
            # Finds a random value within each of the initialisation ranges
            x_init = np.random.uniform(self._init_range_x[0], self._init_range_x[1])
            y_init = np.random.uniform(self._init_range_y[0], self._init_range_y[1])
            coord = np.array([x_init, y_init])  # Coordinate is expressed as a NumPy array of size 2
        return coord 

    def grad_min(self, alpha):
        """Gradient simultaneous minimisation method for 2 dimensions.

        Follows the steepest descent in gradient towards the minimum. This is done by calculating the gradient using a forward
        difference scheme, and taking a small step α in the direction opposite the gradient (as the gradient is perpendicular to the local contour line).
        However, α was scaled so that it has an equivalent relative magnitude in both coordinate directions, for optimal efficiency. 
        The coordinate is updated with each step taken, and iterations occur until the convergence condition is satisfied.

        Args:
            alpha: Size of step taken for each iteration of the descent.
        """
        self._iterations = 0  # Iteration counter
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_coord = self.gen_start_pt()  # Generating starting position
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        self._mins_list.append(self._prev_coord)
        threshold = 1e-6
        h = 1e-6  # Step size for finite differencing (in this case the central-difference scheme)
        # As the mass is smaller in size than the mixing angle, we scale alpha to have the same relative size for both parameters
        # --> The scaling factor is found using the ratio between the means of the two initialisation ranges
        scaling = np.mean(self._init_range_x) / np.mean(self._init_range_y)  
        alpha_x = alpha
        alpha_y = alpha_x / scaling 
        alpha_vec = np.array([alpha_x, alpha_y])  # Alpha is expressed as a vector

        while not self._minimum_found:
            # Finding the gradient vector d, using a central differencing scheme
            d = np.empty(2)
            d[0] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1]) - self.calc_nll(self._prev_coord[0] - h, self._prev_coord[1])) / (2 * h)
            d[1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h) - self.calc_nll(self._prev_coord[0], self._prev_coord[1] - h)) / (2 * h)
            new_coord = self._prev_coord - (alpha_vec * d)  # Calculation of new coordinate vector
            self._mins_list.append(new_coord)
            # print(new_coord)
            if self._iterations == 0:
                # No need to calculate relative difference for the first iteration
                self._prev_coord = new_coord  # Updating the coordinate for the next iteration
            else:
                # Calculation of relative difference in each direction between successive minima
                rel_diff_x = abs(self._prev_coord[0] - new_coord[0]) / self._prev_coord[0]
                rel_diff_y = abs(self._prev_coord[1] - new_coord[1]) / self._prev_coord[1]
                if rel_diff_x < threshold and rel_diff_y < threshold:
                    # Convergence condition: If both x- and y- relative differences are below the threshold (less than 0.0001% of previous minimum),
                    # then triggers the 'minimum_found' flag and exits the loop after this iteration
                    self._minimum_found = True
                    self._min = new_coord  # Saving minimum
                    self._nll_min = self.calc_nll(new_coord[0], new_coord[1])  # Calculating and saving the minimum NLL value at this minimum
                else:
                    self._prev_coord = new_coord  # Updating the coordinate for the next iteration if convergence condition is not met
            
            self._iterations += 1  # Incrementing iteration counter by 1
        
        return self._min  # Returning the coordinate vector that corresponds to the minimum function value

    def newton_min(self, alpha):
        """Newton simultaneous minimisation method for 2 dimensions.

        Takes the local curvature into account at each step for minimisation, by calculating the Hessian and multiplying it by the 
        gradient vector to find the descent vector for each iteration.
        However, the step size used for the central-difference scheme, α, was scaled so that it has an equivalent relative magnitude
        in both coordinate directions, for optimal efficiency. 
        The coordinate is updated with each step taken, and iterations occur until the convergence condition is satisfied.

        Args:
            alpha: Size of step used in central-difference scheme.
        """
        self._iterations = 0  # Iteration counter
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_coord = self.gen_start_pt()  # Generating starting position
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        self._mins_list.append(self._prev_coord)
        threshold = 1e-6
        # As the mass is smaller in size than the mixing angle, we scale alpha to have the same relative size for both parameters.
        # --> The scaling factor is found using the ratio between the means of the two initialisation ranges
        scaling = np.mean(self._init_range_x) / np.mean(self._init_range_y)
        alpha_x = alpha
        alpha_y = alpha_x / scaling 
        alpha_vec = np.array([alpha_x, alpha_y])
        while not self._minimum_found:
            # Finding the gradient vector using central differencing
            grad = np.empty(2)
            grad[0] = (self.calc_nll(self._prev_coord[0] + alpha_x, self._prev_coord[1]) - self.calc_nll(self._prev_coord[0] - alpha_x, self._prev_coord[1])) / (2 * alpha_x)
            grad[1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + alpha_y) - self.calc_nll(self._prev_coord[0], self._prev_coord[1] - alpha_y)) / (2 * alpha_y)
            hessian = np.empty((2,2))  # Initialising the Hessian matrix
            # Calculating each element of the Hessian matrix using central-difference approximation
            hessian[0,0] = ((-1 * self.calc_nll(self._prev_coord[0] + 2 * alpha_x, self._prev_coord[1])) + (16 * self.calc_nll(self._prev_coord[0] + alpha_x, self._prev_coord[1])) \
                            - (30 * self.calc_nll(self._prev_coord[0], self._prev_coord[1])) + (16 * self.calc_nll(self._prev_coord[0] - alpha_x, self._prev_coord[1])) \
                            - self.calc_nll(self._prev_coord[0] - 2 * alpha_x, self._prev_coord[1])) / 12 * (alpha_x ** 2)
            hessian[1,1] = ((-1 * self.calc_nll(self._prev_coord[0], self._prev_coord[1] + 2 * alpha_y)) + (16 * self.calc_nll(self._prev_coord[0], self._prev_coord[1] + alpha_y)) \
                            - (30 * self.calc_nll(self._prev_coord[0], self._prev_coord[1])) + (16 * self.calc_nll(self._prev_coord[0], self._prev_coord[1] - alpha_y)) \
                            - self.calc_nll(self._prev_coord[0], self._prev_coord[1] - 2 * alpha_y)) / 12 * (alpha_y ** 2)
            hessian[0,1] = (self.calc_nll(self._prev_coord[0] + alpha_x, self._prev_coord[1] + alpha_y) - self.calc_nll(self._prev_coord[0] + alpha_x, self._prev_coord[1] - alpha_y) \
                            - self.calc_nll(self._prev_coord[0] - alpha_x, self._prev_coord[1] + alpha_y) - self.calc_nll(self._prev_coord[0] - alpha_x, self._prev_coord[1] - alpha_y)) / (4 * alpha_x * alpha_y) 
            hessian[1,0] = hessian[0,1]
            # print(hessian)
            # hessian = grad[:, np.newaxis] * (np.eye(2) - grad[np.newaxis, :])
            # Calculating the next coordinate step
            new_coord = self._prev_coord - (np.matmul(np.linalg.inv(hessian), grad))
            self._mins_list.append(new_coord)  # Appending new coordinate to list
            # print(alpha_vec * np.matmul(np.linalg.inv(hessian), grad))
            # print('curcoord', self._prev_coord, 'nextcoord', new_coord, 'grad', grad)
            print(new_coord)
            # print('inv hessian', np.linalg.inv(hessian))
            # print(new_coord)
            if self._iterations == 0: 
                # No need to calculate relative difference for the first iteration
                self._prev_coord = new_coord  # Updating the coordinate for the next iteration
            else:
                # Calculation of relative difference in each direction between successive minima
                rel_diff_x = abs(self._prev_coord[0] - new_coord[0]) / self._prev_coord[0]
                rel_diff_y = abs(self._prev_coord[1] - new_coord[1]) / self._prev_coord[1]
                if rel_diff_x < threshold and rel_diff_y < threshold:
                    # Convergence condition: If both x- and y- relative differences are below the threshold (less than 0.0001% of previous minimum),
                    # then triggers the 'minimum_found' flag and exits the loop after this iteration
                    self._minimum_found = True
                    self._min = new_coord  # Saving minimum
                    self._nll_min = self.calc_nll(new_coord[0], new_coord[1])  # Calculating and saving the minimum NLL value at this minimum
                    # if new_coord.all() == self._prev_coord.all():  
                    #     self._iterations -= 1   # If coordinate values from subsequent iterations are the same, don't need to count current iteration
                else:
                    self._prev_coord = new_coord  # Updating the coordinate for the next iteration if convergence condition is not met

            self._iterations += 1  # Incrementing iteration counter by 1

        return self._min  # Returning the coordinate vector that corresponds to the minimum function value


    def quasi_newton_min(self, alpha):
        """Quasi-Newton simultaneous minimisation method for 2 dimensions.

        A less computationally intensive approximation of the Newton method, which uses the local gradient to approximate the inverse Hessian.
        However, the step size used, α, was scaled so that it has an equivalent relative magnitude in both coordinate directions, for optimal efficiency. 
        The coordinate is updated with each step taken, and iterations occur until the convergence condition is satisfied.

        Args:
            alpha: Size of step used in central-difference scheme.
        """
        self._iterations = 0  # Iteration counter
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_coord = self.gen_start_pt()  # Generating starting position
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        self._mins_list.append(self._prev_coord)
        threshold = 1e-6
        h = 1e-6  # Step size for finite differencing (in this case the central-difference scheme)
        # As the mass is smaller in size than the mixing angle, we scale alpha to have the same relative size for both parameters.
        # --> The scaling factor is found using the ratio between the means of the two initialisation ranges
        scaling = np.mean(self._init_range_x) / np.mean(self._init_range_y)
        alpha_x = alpha
        alpha_y = alpha_x / scaling 
        alpha_vec = np.array([alpha_x, alpha_y])  # Alpha is expressed as a vector
        G = np.identity(2)  # G (inverse Hessian approximation) is equal to the identity matrix for the first iteration.
        self._grad = np.empty(2)  # Initialising the gradient vector
        while not self._minimum_found:
            
            if self._iterations == 0:
                # Need to compute the gradient vector for the first iteration (grad is updated during the calculations of updating G for later iterations)
                # --> Central difference scheme used to compute gradient vector
                self._grad[0] = (self.calc_nll(self._prev_coord[0] + h, self._prev_coord[1]) - self.calc_nll(self._prev_coord[0] - h, self._prev_coord[1])) / (2 * h)
                self._grad[1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + h) - self.calc_nll(self._prev_coord[0], self._prev_coord[1] - h)) / (2 * h)

            new_coord = self._prev_coord - (alpha_vec * np.matmul(G, self._grad))  # Calculating the new coordinate for this step
            self._mins_list.append(new_coord)  # Appending new coordinate to list
            print(new_coord)
            if self._iterations == 0:  # No need to check for convergence if first iteration
                self._prev_coord = new_coord  # Updating the coordinate for the next iteration
            else:
                # Calculation of relative difference in each direction between successive minima
                rel_diff_x = abs(self._prev_coord[0] - new_coord[0]) / self._prev_coord[0]
                rel_diff_y = abs(self._prev_coord[1] - new_coord[1]) / self._prev_coord[1]
                if rel_diff_x < threshold and rel_diff_y < threshold:
                    # Convergence condition: If both x- and y- relative differences are below the threshold (less than 0.00001% of previous minimum),
                    # then triggers the 'minimum_found' flag and exits the loop after this iteration
                    self._minimum_found = True
                    self._min = new_coord  # Saving minimum
                    self._nll_min = self.calc_nll(new_coord[0], new_coord[1])  # Calculating and saving the minimum NLL value at this minimum
                else:
                    # If the initial convergence condition is not fulfilled
                    delta_n = new_coord - self._prev_coord  # Calculating the vector δ_n
                    new_grad = np.empty(2)
                    new_grad[0] = (self.calc_nll(new_coord[0] + h, new_coord[1]) - self.calc_nll(new_coord[0] - h, new_coord[1])) / (2 * h)
                    new_grad[1] = (self.calc_nll(new_coord[0], new_coord[1] + h) - self.calc_nll(new_coord[0], new_coord[1] - h)) / (2 * h)            

                    gamma_n = new_grad - self._grad  # Calculating the vector γ_n
                    gd_prod = np.dot(gamma_n, delta_n)  # Vector dot product of (γ_n, δ_n)
                    # Alternative convergence condition - if gamma_n * delta_n is equal to zero
                    # This also means that the step size is sufficiently small (G also cannot be updated due to division by zero)
                    if gd_prod == 0:
                        self._minimum_found = True
                        self._min = new_coord
                        self._nll_min = self.calc_nll(new_coord[0], new_coord[1])
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

    def quasi_LMA_min(self, alpha):
        """Quasi-Newton simultaneous minimisation method for 2 dimensions.

        A less computationally intensive approximation of the Newton method, which uses the local gradient to approximate the inverse Hessian.
        However, the step size used, α, was scaled so that it has an equivalent relative magnitude in both coordinate directions, for optimal efficiency. 
        The coordinate is updated with each step taken, and iterations occur until the convergence condition is satisfied.

        Args:
            alpha: Size of step used in central-difference scheme.
        """
        self._iterations = 0  # Iteration counter
        self._minimum_found = False  # Flag for the minimum being found
        self._prev_coord = self.gen_start_pt()  # Generating starting position
        self._mins_list = []  # Initialising list of minima (for plotting purposes)
        self._mins_list.append(self._prev_coord)
        # As the mass is smaller in size than the mixing angle, we scale alpha to have the same relative size for both parameters.
        # --> The scaling factor is found using the ratio between the means of the two initialisation ranges
        scaling = np.mean(self._init_range_x) / np.mean(self._init_range_y)
        alpha_x = alpha
        alpha_y = alpha_x / scaling 
        alpha_vec = np.array([alpha_x, alpha_y])  # Alpha is expressed as a vector
        G = np.identity(2)  # G (inverse Hessian approximation) is equal to the identity matrix for the first iteration.
        self._grad = np.empty(2)  # Initialising the gradient vector
        while not self._minimum_found:
            
            if self._iterations == 0:
                # Need to compute the gradient vector for the first iteration (grad is updated during the calculations of updating G for later iterations)
                # --> Central difference scheme used to compute gradient vector
                self._grad[0] = (self.calc_nll(self._prev_coord[0] + alpha_x, self._prev_coord[1]) - self.calc_nll(self._prev_coord[0] - alpha_x, self._prev_coord[1])) / (2 * alpha_x)
                self._grad[1] = (self.calc_nll(self._prev_coord[0], self._prev_coord[1] + alpha_y) - self.calc_nll(self._prev_coord[0], self._prev_coord[1] - alpha_y)) / (2 * alpha_y)

            new_coord = self._prev_coord - (alpha_vec * np.matmul(G, self._grad))  # Calculating the new coordinate for this step
            self._mins_list.append(new_coord)  # Appending new coordinate to list
            if self._iterations == 0:  # No need to check for convergence if first iteration
                self._prev_coord = new_coord  # Updating the coordinate for the next iteration
            else:
                # Calculation of relative difference in each direction between successive minima
                rel_diff_x = abs(self._prev_coord[0] - new_coord[0]) / self._prev_coord[0]
                rel_diff_y = abs(self._prev_coord[1] - new_coord[1]) / self._prev_coord[1]
                if rel_diff_x < threshold and rel_diff_y < threshold:
                    # Convergence condition: If both x- and y- relative differences are below the threshold (less than 0.00001% of previous minimum),
                    # then triggers the 'minimum_found' flag and exits the loop after this iteration
                    self._minimum_found = True
                    self._min = new_coord  # Saving minimum
                    self._nll_min = self.calc_nll(new_coord[0], new_coord[1])  # Calculating and saving the minimum NLL value at this minimum
                else:
                    # If the initial convergence condition is not fulfilled
                    delta_n = new_coord - self._prev_coord  # Calculating the vector δ_n
                    new_grad = np.empty(2)
                    new_grad[0] = (self.calc_nll(new_coord[0] + alpha_x, new_coord[1]) - self.calc_nll(new_coord[0] - alpha_x, new_coord[1])) / (2 * alpha_x)
                    new_grad[1] = (self.calc_nll(new_coord[0], new_coord[1] + alpha_y) - self.calc_nll(new_coord[0], new_coord[1] - alpha_y)) / (2 * alpha_y)
                    gamma_n = new_grad - self._grad  # Calculating the vector γ_n
                    gd_prod = np.dot(gamma_n, delta_n)  # Vector dot product of (γ_n, δ_n)
                    # Alternative convergence condition - if gamma_n * delta_n is equal to zero
                    # This also means that the step size is sufficiently small (G also cannot be updated due to division by zero)
                    if gd_prod == 0:
                        self._minimum_found = True
                        self._min = new_coord
                        self._nll_min = self.calc_nll(new_coord[0], new_coord[1])
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
    def x_iters(self):
        return self._x_iters
    
    @property
    def y_iters(self):
        return self._y_iters

    @property
    def mins_list(self):
        return self._mins_list
