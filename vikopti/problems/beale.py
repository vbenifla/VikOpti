import numpy as np
from vikopti.core.problem import Problem


class Beale(Problem):
    """
    Class representing the Beale function optimization problem.
    """

    def __init__(self):
        """
        Constructs the problem object and set the different attributes.
        """
        super().__init__(2, 1, 0, True)

        # set problem's name
        self.name = "Beale function"

        # set variables and boundaries
        self.var = ['x', 'y']
        self.bounds = np.array([[-4.5, 4.5], [-4.5, 4.5]])

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

        a = 1.5 - x[0] + x[0] * x[1]
        b = 2.25 - x[0] + x[0] * (x[1]**2)
        c = 2.625 - x[0] + x[0] * (x[1]**3)
        f = a**2 + b**2 + c**2

        return np.array([-f])
