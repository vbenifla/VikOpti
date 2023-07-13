import numpy as np
from vikopti.core.problem import Problem


class Ackley(Problem):
    """
    Class representing the Ackley function optimization problem.
    """

    def __init__(self):
        """
        Constructs the problem object and set the different attributes.
        """
        super().__init__(2, 1, 0, True)

        # set problem's name
        self.name = "Ackley function"

        # set variables and boundaries
        self.var = ['x', 'y']
        self.bounds = np.array([[-5, 5], [-5, 5]])

    def func(self, x: np.ndarray):
        """
        Objectives and constraints functions.

        Parameters
        ----------
        x : np.ndarray
            variables.

        Returns
        -------
        np.ndarray
            Objectives and constraints values.
        """

        a = np.sqrt((x[0]**2 + x[1]**2) / 2) / 5
        b = (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])) / 2
        f = -20 * np.exp(-a) - np.exp(b) + np.e + 20

        return np.array([-f])
