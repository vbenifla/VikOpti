import numpy as np
from vikopti.core.problem import Problem


class GL(Problem):
    """
    Class representing the Gomez and Levy function optimization problem.
    """

    def __init__(self):
        """
        Constructs the problem object and set the different attributes.
        """
        super().__init__(2, 1, 1, True)

        # set problem's name
        self.name = "Gomez and Levy function"

        # set variables and boundaries
        self.var = ['x', 'y']
        self.bounds = np.array([[-1, 0.75], [-1, 1]])

        # add constraints
        self.add_constraint('inf', 1.5)

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
            Objectives and constraints values
        """

        a = 4 * x[0]**2 - 4 * x[1]**2
        b = 4 * x[1]**4 - 2.1 * x[0]**4
        c = (x[0] ** 6) / 3
        d = x[0] * x[1]
        f = a + b + c + d

        a = 2 * np.sin(2 * np.pi * x[1]) ** 2
        b = np.sin(4 * np.pi * x[0])
        g = a - b

        return np.array([-f, g])
