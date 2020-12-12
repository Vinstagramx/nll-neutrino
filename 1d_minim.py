# Importing relevant package
import numpy as np

class Minimise1D():
    """Class which carries out 1-D minimisation

    """
    def __init__(self, nll_vals, param_arr, init_guess):
        """Initialisation of the Minimise1D class.

        Input arguments are saved internally within the class, and used when the class methods are called.
        Ensures that NLL and parameter arrays are of the same length.

        Args:
            nll_vals: NumPy array containing values of the NLL for each value of the parameter.
            param_arr: Array containing values of the corresponding parameter for which the NLL is to be minimised.
            init_guess: Initial guess for the parameter value which corresponds to the NLL minimum.
        
        Raises:
            AttributeError: If the input arrays are not of the same length.
        """
        # Checking for input errors
        if len(nll_vals) != len(param_arr):
            raise AttributeError("Input NLL and parameter arrays must be of equal length!")
        if not isinstance(nll_vals, np.ndarray) and isinstance(param_arr, np.ndarray):
            raise TypeError("Please ensure that energies and event rates are in the form of NumPy arrays!")

        # Saving the relevant data as private member variables for later use
        self._nll = nll_vals
        self._params = param_arr
        self._init_guess = init_guess
    
    def para_min(self):
        """Parabolic minimiser method to find minimum of NLL.
        """

        

