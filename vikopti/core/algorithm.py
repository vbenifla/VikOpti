import time
import numpy as np
import multiprocessing as mp
from vikopti.core.problem import Problem
from vikopti.core.results import Results


class Algorithm:
    """
    Class representing an algorithm.
    """

    def __init__(self, problem: Problem, display: bool = True, save: bool = False, n_proc: int = 1):
        """
        Constructs the algorithm object and set the different attributes.

        Parameters
        ----------
        problem : Problem
            problem considered.
        display : bool, optional
            option to display algorithm's outputs in the console, by default True.
        save : bool, optional
            option to save algorithm's results, by default False.
        n_proc : int, optional
            number of worker processes to use for multi-processing, by default 1.
        """

        # set problem
        self.problem = problem

        # set display and save options
        self.display = display
        self.save = save

        # set number of worker processes
        self.n_proc = n_proc

    def start(self):
        """
        Start the algorithm.
        """

        # check if enough resources is available and start multi-processing pool
        self.n_proc = self.n_proc if self.n_proc < mp.cpu_count() - 1 else 2
        self.pool = mp.Pool(self.n_proc)

        # initialize results and evaluation counter
        self.results = Results(self)
        self.f_eval = 0

        # print algorithm's and problem's summary
        if self.display:
            self.print()
            self.problem.print()
            print("###################")
            print("####### Run #######")
            print("###################")

        # start timer
        self.st = time.time()

    def stop(self):
        """
        Stop the algorithm.
        """

        # stop timer
        self.et = time.time()

        # close multi-processing pool
        self.pool.close()

        # wait for worker processes to terminate
        self.pool.join()

        # set results
        self.results.run_time = round(self.et - self.st)
        self.results.f_eval = self.f_eval

        # save algorithm's results
        if self.save:
            self.results.save()

        # print and plot algorithm's results
        if self.display:
            self.results.print()

    def evaluate(self, x: np.ndarray):
        """
        Evaluate the algorithm's problem.

        Parameters
        ----------
        x : np.ndarray
            variables.

        Returns
        -------
        obj : np.ndarray
            objectives values.
        const : np.ndarray
            constraints values.
        """

        # run batch
        res = np.vstack(self.pool.map(self.problem.func, x))

        # get objective and constraints
        obj = res[:, :self.problem.n_obj]
        const = res[:, self.problem.n_obj:self.problem.n_obj + self.problem.n_const]

        # update counter
        self.f_eval += len(x)

        return obj, const

    def __getstate__(self):
        """
        Some function to prevent bugs wit pool as an attributes.
        """

        self_dict = self.__dict__.copy()
        del self_dict["pool"]

        return self_dict

    def __setstate__(self, state):
        """
        Some function to prevent bugs wit pool as an attributes.
        """

        self.__dict__.update(state)
