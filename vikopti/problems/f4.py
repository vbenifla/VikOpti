import numpy as np
from vikopti.core.problem import Problem


class F4(Problem):
    """
    Class representing the F4 function.
    From: M. Hall. 2012. A Cumulative Multi-Niching Genetic Algorithm for Multimodal Function Optimization.
    """

    def __init__(self):
        """
        Constructs the problem object and set the different attributes.
        """
        super().__init__(2, 1, 0, True)

        # set problem's name
        self.name = "F4 function"

        # set variables and boundaries
        self.var = ['x', 'y']
        self.bounds = np.array([[-40, 40], [-40, 40]])

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

        A = [-20, 40, 0]
        B = [-20, -30, 30]
        H = [0.7, 1, 1.5]
        W = [0.02, 0.08, 0.01]
        f = 0
        for i in range(len(A)):
            f += H[i] / (1 + W[i] * ((x[0] - A[i]) ** 2 + (x[1] - B[i]) ** 2))

        return np.array([f])
