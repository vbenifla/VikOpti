import numpy as np
from vikopti.core.problem import Problem


class MCC(Problem):
    """
    Class representing the McCormick function optimization problem.

    """

    def __init__(self):
        """
        Constructs the problem object and set the different attributes.
        """
        super().__init__(2, 1, 0, True)

        # set problem's name
        self.name = "McCormick function"

        # set variables and boundaries
        self.var = ['x', 'y']
        self.bounds = np.array([[-1.5, 4], [-3, 4]])

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

        a = np.sin(x[0] + x[1])
        b = (x[0] - x[1])**2
        c = -1.5 * x[0] + 2.5 * x[1] + 1
        f = a + b + c

        return np.array([-f])
