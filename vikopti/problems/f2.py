import numpy as np
from vikopti.core.problem import Problem


class F2(Problem):
    """
    Class representing the F2 function.
    From: M. Hall. 2012. A Cumulative Multi-Niching Genetic Algorithm for Multimodal Function Optimization.
    """

    def __init__(self):
        """
        Constructs the problem object and set the different attributes.
        """
        super().__init__(1, 1, 0, True)

        # set problem's name
        self.name = "F2 function"

        # set variables and boundaries
        self.var = ['x']
        self.bounds = np.array([[0, 0.9]])

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

        a = np.sin(5.1 * np.pi * x[0] + 0.5) ** 6
        b = - (4 / 0.64) * np.log(2) * (x[0] - 0.0667) ** 2
        f = a * np.exp(b)

        return np.array([f])
