import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
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

        # initialize evaluation counter
        self.f_eval = 0

        # initialize results
        self.results = Results(self.problem, self.save)

        # check if enough resources is available
        self.n_proc = self.n_proc if self.n_proc < mp.cpu_count() - 1 else 2

        # start multi-processing pool
        self.pool = mp.Pool(self.n_proc)

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

        # print and plot algorithm's results
        if self.display:
            self.results.print()
            self.results.plot()
            plt.show()

        # save algorithm's results
        if self.save:
            self.results.save()

    def evaluate(self, x: np.ndarray):
        """
        Evaluate the algorithm's problem.

        Parameters
        ----------
        x : np.ndarray
            design variables to evaluate.

        Returns
        -------
        obj : np.ndarray
            evaluated objective.
        const : np.ndarray
            evaluated constraints.
        """

        # get batch size
        batch_size = len(x)

        # run batch
        res = self.pool.map(self.problem.func, x)

        # initialize arrays to return
        obj = np.zeros((batch_size, self.problem.n_obj))
        const = np.zeros((batch_size, self.problem.n_const))

        # post-process results
        for i in range(batch_size):
            obj[i] = res[i][0]

            if self.problem.n_const > 0:
                const[i] = res[i][1]

        # update counter
        self.f_eval += batch_size

        return obj, const

    def __getstate__(self):

        self_dict = self.__dict__.copy()
        del self_dict["pool"]

        return self_dict

    def __setstate__(self, state):

        self.__dict__.update(state)

    def compute_penalty(self, const: np.ndarray):
        """
        Compute the penalty from constraint.

        Parameters
        ----------
        g : np.ndarray
            constraints.

        Returns
        -------
        penalty : np.ndarray
            penalty values.
        """

        # set array containing penalty
        penalty = np.zeros(len(const))

        # loop on constraints
        for i in range(self.problem.n_const):
            # compute absolute constraint violation
            violation = np.abs(self.problem.constraint[i].limit - const[:, i])

            # apply penalty only where the constraint is not respected
            if self.problem.constraint[i].type == "inf":
                penalty += np.where(const[:, i] > self.problem.constraint[i].limit, violation, 0)

            elif self.problem.constraint[i].type == "sup":
                penalty += np.where(const[:, i] < self.problem.constraint[i].limit, violation, 0)

        return penalty

    def compute_fitness(self):
        """
        Compute and scale the fitness of the current population.
        """

        # find feasible solution in the current population
        feasible = self.p[:self.pop_size] == 0

        # find worst feasible solution in the current population and get its objective
        if len(feasible) != 0:
            f_worst = np.min(self.obj[:self.pop_size][feasible])
        else:
            f_worst = 0

        # compute fitness of the current population
        self.f[:self.pop_size][feasible] = self.obj[:self.pop_size, 0][feasible]
        self.f[:self.pop_size][~feasible] = f_worst - self.p[:self.pop_size][~feasible]

        # scale fitness of the current population
        f_min = self.f[:self.pop_size].min()
        f_max = self.f[:self.pop_size].max()
        self.f[:self.pop_size] = (self.f[:self.pop_size] - f_min) / (f_max - f_min)

    def sbx(self, x_p: np.ndarray, eta: float = 15):
        """
        Perform Simulated-Binary-Crossover for real-coded GA. Fromula from K. Deb, K. Sindhya, and T. Okabe. 2007.
        Self-adaptive simulated binary crossover for real-parameter optimization.

        Parameters
        ----------
        x_p : np.ndarray
            parents's design variables.
        eta : float, optional
            distribution index which is any non-negative real number, by default 15.

        Returns
        -------
        x_o : np.ndarray
            offsprings's design variables.
        """

        # set array containing offsprings's design variables
        x_o = np.zeros((2, self.problem.n_var))

        # create a random values for each individual
        rand = np.random.random(2)

        # compute delta
        x_max = np.max(x_p, axis=0)
        x_min = np.min(x_p, axis=0)
        delta = x_max - x_min
        delta[delta == 0] = 1e-11

        # compute beta
        beta = 1.0 + (2.0 * (x_min - self.problem.bounds[:, 0]) / delta)

        # compute betaq
        alpha = 2.0 - np.power(beta, -(eta + 1.0))
        mask, mask_not = (rand[0] <= (1.0 / alpha)), (rand[0] > (1.0 / alpha))
        betaq = np.zeros(mask.shape)
        betaq[mask] = np.power((rand[0] * alpha), (1.0 / (eta + 1.0)))[mask]
        betaq[mask_not] = np.power((1.0 / (2.0 - rand[0] * alpha)), (1.0 / (eta + 1.0)))[mask_not]

        # compute first offspring's design variables
        x_o[0] = 0.5 * ((x_max + x_min) - betaq * delta)

        # compute beta
        beta = 1.0 + (2.0 * (self.problem.bounds[:, 1] - x_max) / delta)

        # compute betaq
        alpha = 2.0 - np.power(beta, -(eta + 1.0))
        mask, mask_not = (rand[1] <= (1.0 / alpha)), (rand[1] > (1.0 / alpha))
        betaq = np.zeros(mask.shape)
        betaq[mask] = np.power((rand[1] * alpha), (1.0 / (eta + 1.0)))[mask]
        betaq[mask_not] = np.power((1.0 / (2.0 - rand[1] * alpha)), (1.0 / (eta + 1.0)))[mask_not]

        # compute second offspring's design variables
        x_o[1] = 0.5 * ((x_max + x_min) + betaq * delta)

        return x_o

    def plot_population(self):

        # if the problem is 2D
        if self.problem.n_var == 2:

            # plot pb
            fig, ax = self.problem.plot_f_contour()

            # plot population
            plt.scatter(self.x[:self.pop_size, 0], self.x[:self.pop_size, 1], color='black')
