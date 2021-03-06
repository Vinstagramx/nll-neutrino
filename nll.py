# Importing relevant package
import numpy as np

class NLL():
    """Class which calculates the Negative Log Likelihood. Also uses the oscillation parameters to produce the oscillated event rate prediction.
    
    An instance of this class is initialised using the midpoints of the energy bins, the simulated event rates, and various oscillation parameters.
    Contains methods to calculate the survival probability P(ν_µ --> ν_µ) and the oscillated event rate λ for each 'bin' of energy. 
    The NLL can then be found only after these are calculated.
    --> needs edit after section 5
    """

    def __init__(self, energies, event_rates, obs_events, mix_ang = np.pi/4, distance = 295, sq_mass_diff = 2.4e-3, cross_sec = None):
        """Initialisation of the NLL class.

        Input arguments are saved internally within the class, and used when the class methods are called.
        Ensures that the energy midpoints and the simulated event rates are inputted as NumPy arrays of equal length.

        Args:
            energies: NumPy array containing the muon neutrino energies for which the oscillated event rates will be generated.
            event_rates: NumPy array containing prediction of unoscillated event rates from simulation.
            obs_events: NumPy array containing the observed number of neutrino events for each energy 'bin'.
            mix_ang: Mixing angle, in radians (default value pi/4).
            distance: Distanced travelled by the muon neutrino, in km.
            sq_mass_diff: Difference between the squared masses of the two neutrinos, in eV^2.
            cross_sec: Constant of proportionality (α) of cross section increase with energy (default set to None).

        Raises:
            AttributeError: If the NumPy arrays 'energies', 'event_rates' and 'obs_events' are not of the same length.
            TypeError: If the energy midpoints and the simulated event rates are not inputted in the form of NumPy arrays.
        """
        # Checking for input errors
        if len(energies) != len(event_rates) != len(obs_events):
            raise AttributeError("Input arrays must be of equal length!")
        if not isinstance(energies, np.ndarray) and isinstance(event_rates, np.ndarray):
            raise TypeError("Please ensure that energies and event rates are in the form of NumPy arrays!")

        # Saving the relevant data as private member variables for later use
        self._energies = energies
        self._event_rates = event_rates
        self._obs_events = obs_events
        self._mix_ang = mix_ang
        self._dist = distance
        self._sq_mass_diff = sq_mass_diff
        self._cross_sec = cross_sec

        # Initialising flags for later checks (to see if methods have been carried out)
        self._probs_found = False
        self._lambdas_found = False

    def surv_prob(self):
        """Calculates the survival probabilities (P(ν_µ --> ν_µ)) of the muon neutrino.

        Using the array of energies given during initialisation, generates an array of survival probabilities
        by using Equation 1 from the project brief document. Returns this array of probabilities.

        Returns:
            self._probs: Array of survival (non-oscillation) probabilities.
        """
        prob_list = np.empty(len(self._energies))  # Initialising an empty (placeholder) NumPy array for probability values

        # Calculating the survival probability for each energy 'bin'
        for i, val in enumerate(self._energies):
            osc_prob = (np.sin(2 * self._mix_ang) ** 2) * (np.sin((1.267 * self._sq_mass_diff * self._dist)/ val) ** 2)
            surv_prob = 1 - osc_prob 
            prob_list[i] = surv_prob  # Assigns the calculated probability value with the corresponding index
        
        self._probs = np.array(prob_list)  # Saving the probabilities within the class for later use
        self._probs_found = True 
        return self._probs  # Returns the survival probabilities (in the form of a NumPy array)
    
    def calc_lambda(self):
        """Calculates the oscillated event rates (λ) for each energy bin from the simulated event rates provided.

        Takes the element-wise product of the survival probability and the simulated event rate NumPy arrays to calculate
        λ for each bin. The resulting array of oscillated event rates is then returned.
        Checks that the survival probabilities have been found before calculation.
        Takes into account the cross-sectional scaling factor (α) if it is inputted during initialisation. 

        Returns:
            self._lambdas: Array of oscillated event rates.
        
        Raises:
            AttributeError: If the survival probabilities have not yet been calculated.
        """
        # Checking that the survival probabilities have been found
        if not self._probs_found:
            raise AttributeError('Please calculate survival probabilities using surv_prob() before finding oscillated event rates.')
        lambda_u = self._probs * self._event_rates  # Multiplication to find list of λ

        if self._cross_sec is not None:
            # Taking into account the constant of proportionality/scaling factor of cross section increase with energy
            lambda_u = lambda_u * self._cross_sec * self._energies 

        self._lambdas = lambda_u  # Saving the array of λs within the class for later use
        self._lambdas_found = True 
        return self._lambdas  # Returns the oscillated event rates (in the form of a NumPy array)

    def find_nll(self):
        """Calculates the Negative Log Likelihood (NLL) of the probability distribution function, given the oscillation parameters supplied.

        The NLL is found for the given set of oscillation parameters in this class instance, using Equation 6 from the project brief document.
        Uses the oscillated event rates calculated previously in the class instance, and the observed neutrino event numbers
        to initialise the class. Checks that the oscillated event rates have been calculated previously.

        Returns:
            self._NLL: Value of negative log likelihood.
        Raises:
            AttributeError: If oscillated event rates have not yet been calculated.
        """
        # Checking that oscillated event rates have been found
        if not self._lambdas_found:
            raise AttributeError('Please calculate oscillated event rates using calc_lambda() before finding the NLL.')
        sum = 0  # Initial value of sum
        # Adding on the sum terms calculated for each energy 'bin'
        for i, val in enumerate(self._lambdas):
            if self._obs_events[i] == 0:
                sum_term = val  # Remainder of the sum term becomes zero (if statement used to prevent np.log(0) calculation)
            else: 
                sum_term = val - self._obs_events[i] + (self._obs_events[i] * np.log(self._obs_events[i]/val))
            sum += sum_term
        NLL = sum

        self._NLL = NLL  # Saving the NLL value within the class for later use
        return self._NLL  # Returns the NLL value



