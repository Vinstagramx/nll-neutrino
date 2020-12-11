import numpy as np

class EventRate():
    """Class which varies the oscillation parameters to produce the oscillated event rate prediction
    (Used in sections 3.1, and 5 - cross-section)
    """

    def __init__(self, energies, event_rates, mix_ang = np.pi/4, distance = 295, sq_mass_diff = 2.4e-3):
        """Initialisation of the EventRate class.

        Input arguments are saved internally within the class, and used when the class methods are called.

        Args:
            energies: List/NumPy array containing the muon neutrino energies for which the oscillated event rates will be generated.
            mix_ang: Mixing angle, in radians (default value pi/4).
            distance: Distanced travelled by the muon neutrino, in km.
            sq_mass_diff: Difference between the squared masses of the two neutrinos, in eV^2.
        """
        # Checking for errors
        if len(energies) != len(event_rates):
            raise AttributeError("Energy and simulated event rate arrays must be of equal length!")
        if not isinstance(energies, np.array) and isinstance(event_rates, np.array):
            raise TypeError("Please ensure that energies and event rates are in the form of NumPy arrays!")

        self._energies = energies
        self._event_rates = event_rates
        self._mix_ang = mix_ang
        self._dist = distance
        self._sq_mass_diff = sq_mass_diff

    def surv_prob(self):
        """Calculates the survival probabilities of the muon neutrino.

        Using the array of energies given during initialisation, generates an array of survival probabilities.

        Returns:
            prob_arr: Array of survival (non-oscillation) probabilities.
        """
        prob_list = np.zeros(len(self._energies))
        for i in self._energies:
            osc_prob = np.sin(np.sin(2 * self._mix_ang)) * np.sin(np.sin((1.267 * self._sq_mass_diff * self._dist)/ i))
            surv_prob = 1 - osc_prob 
            prob_list[i] = surv_prob
        
        self._probs = np.array(prob_list)
        return self._probs
    
    def calc_lambda(self):
        """Calculates the oscillated event rates from the simulated event rates provided.
        """

        lambda_u = self._probs * self._energies
        self._lambdas = lambda_u
        
        return self._lambdas

    
