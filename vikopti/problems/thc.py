import numpy as np
from vikopti.core.problem import Problem


class THC(Problem):
    """
    Class representing the Three-Hump-Camel function optimization problem.
    """

    def __init__(self):
        """
        Constructs the problem object and set the different attributes.
        """
        super().__init__(2, 1, 0, True)

        # set problem's name
        self.name = "Three-Hump-Camel function"

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

        a = 2 * x[0]**2 + x[1]**2
        b = -1.05 * x[0]**4
        c = (x[0]**6) / 6
        d = x[0] * x[1]
        f = a + b + c + d

        return np.array([-f])
