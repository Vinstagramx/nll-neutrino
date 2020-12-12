# Importing relevant package
import numpy as np

class NLL():
    """Class which calculates the Negative Log Likelihood. Also uses the oscillation parameters to produce the oscillated event rate prediction.
    
    An instance of this class is initialised using the midpoints of the energy bins, the simulated event rates, and various oscillation parameters.
    Contains methods to calculate the survival probability P(ν_µ --> ν_µ) and the oscillated event rate λ for each 'bin' of energy. 
    The NLL can then be found only after these are calculated.
    --> needs edit after section 5
    """

    def __init__(self, energies, event_rates, mix_ang = np.pi/4, distance = 295, sq_mass_diff = 2.4e-3):
        """Initialisation of the EventRate class.

        Input arguments are saved internally within the class, and used when the class methods are called.
        Ensures that the energy midpoints and the simulated event rates are inputted as NumPy arrays of equal length.

        Args:
            energies: NumPy array containing the muon neutrino energies for which the oscillated event rates will be generated.
            event_rates: NumPy array containing prediction of unoscillated event rates from simulation.
            mix_ang: Mixing angle, in radians (default value pi/4).
            distance: Distanced travelled by the muon neutrino, in km.
            sq_mass_diff: Difference between the squared masses of the two neutrinos, in eV^2.

        Raises:
            AttributeError: If the NumPy arrays 'energies' and 'event_rates' are not of the same length.
            TypeError: If the energy midpoints and the simulated event rates are not inputted in the form of NumPy arrays.
        """
        # Checking for errors
        if len(energies) != len(event_rates):
            raise AttributeError("Energy and simulated event rate arrays must be of equal length!")
        if not isinstance(energies, np.ndarray) and isinstance(event_rates, np.ndarray):
            raise TypeError("Please ensure that energies and event rates are in the form of NumPy arrays!")

        # Saving the relevant data as private member variables for later use
        self._energies = energies
        self._event_rates = event_rates
        self._mix_ang = mix_ang
        self._dist = distance
        self._sq_mass_diff = sq_mass_diff

        # Initialising empty member variables for later checks (to see if methods have been carried out)
        self._probs = None
        self._lambdas = None

    def surv_prob(self):
        """Calculates the survival probabilities (P(ν_µ --> ν_µ)) of the muon neutrino.

        Using the array of energies given during initialisation, generates an array of survival probabilities
        by using Equation 1 from the project brief document. Returns this array of probabilities.

        Returns:
            self._probs: Array of survival (non-oscillation) probabilities.
        """
        prob_list = np.zeros(len(self._energies))  # Initialising an empty (placeholder) NumPy array for probability values

        # Calculating the survival probability for each energy 'bin'
        for i, val in enumerate(self._energies):
            osc_prob = (np.sin(2 * self._mix_ang) ** 2) * (np.sin((1.267 * self._sq_mass_diff * self._dist)/ val) ** 2)
            surv_prob = 1 - osc_prob 
            prob_list[i] = surv_prob  # Assigns the calculated probability value with the corresponding index
        
        self._probs = np.array(prob_list)  # Saving the probabilities within the class for later use
        return self._probs  # Returns the survival probabilities (in the form of a NumPy array)
    
    def calc_lambda(self):
        """Calculates the oscillated event rates (λ) for each energy bin from the simulated event rates provided.

        Takes the element-wise product of the survival probability and the energy NumPy arrays to calculate
        λ for each bin. The resulting array of oscillated event rates is then returned.
        Checks that the survival probabilities have been found before calculation.

        Returns:
            self._lambdas: Array of oscillated event rates.
        
        Raises:
            AttributeError: If self._probs is empty - i.e. the survival probabilities have not yet been calculated.
        """
        if self._probs = None:
            raise AttributeError('Please calculate survival probabilities using surv_prob() before finding oscillated event rates.')
        lambda_u = self._probs * self._energies  # Multiplication to find list of λ
        self._lambdas = lambda_u  # Saving the array of λs within the class for later use
        
        return self._lambdas  # Returns the oscillated event rates (in the form of a NumPy array)

    def find_nll(self):
        """Calculates the Negative Log Likelihood (NLL) of the probability distribution function, given the oscillation parameters supplied.

        The NLL is found for the given set of oscillation parameters in this class instance, using Equation 6 from the project brief document.
        Uses the oscillated event rates calculated previously in the class instance, and the simulated unoscillated event rates used
        to initialise the class. Checks that the oscillated event rates have been calculated previously.

        Returns:
            self._NLL: Value of negative log likelihood.
        Raises:

        """
        if self._lambdas = None:
            raise AttributeError('Please calculate oscillated event rates using calc_lambda() before finding the NLL.')
        sum = 0  # Initial value of sum
        # Adding on the sum terms calculated for each energy 'bin'
        for i, val in enumerate(self._lambdas):
            sum_term = val - self._event_rates[i] + (self._event_rates[i] * np.log(self._event_rates[i]/val))
            sum += sum_term

        NLL = sum
        self._NLL = NLL  # Saving the NLL value within the class for later use

        return self._NLL  # Returns the NLL value



