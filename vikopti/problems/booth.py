import numpy as np
from vikopti.core.problem import Problem


class Booth(Problem):
    """
    Class representing the Booth function optimization problem.
    """

    def __init__(self):
        """
        Constructs the problem object and set the different attributes.
        """
        super().__init__(2, 1, 0, True)

        # set problem's name
        self.name = "Booth function"

        # set variables and boundaries
        self.var = ['x', 'y']
        self.bounds = np.array([[-10, 10], [-10, 10]])

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

        a = x[0] + 2 * x[1] - 7
        b = x[1] + 2 * x[0] - 5
        f = a**2 + b**2

        return np.array([-f])
