import numpy as np
from pysampling.sample import sample
from vikopti.core.problem import Problem
from vikopti.core.algorithm import Algorithm
from vikopti.core.utils import compute_penalty, compute_fitness, get_optima, sbx


class VIKGA(Algorithm):
    """
    Class representing a pretty cool genetic algorithm.
        - It is a real-coded GA.
        - The population is cumulative and extends through the generation.
        - It offers different sampling method for the initial population.
        - The crossover operation is performed using Simulated-Binary-Crossover.
        - Crossover parents are selected through a combined fitness and distance-proportionate selection process.
        - The mutation operation is performed randomly.
        - The addition operation prevents unnecessary objective function evaluation.
        - It uses an efficient constraint handling technic through applying penalty functions.
    """

    def __init__(self, problem: Problem, n_min: int = 10, n_max: int = 100,
                 n_gen: int = 100, n_cross: int = 6, n_mute: int = 3):
        """
        Constructs the algorithm object and set the different attributes.

        Parameters
        ----------
        problem : Problem
            problem considered which the algorithm is trying to solve.
        n_min : int, optional
            minimum population size, by default 10.
        n_max : int, optional
            maximum population size, by default 100.
        n_gen : int, optional
            number of generations, by default 100.
        n_cross : int, optional
            number of crossovers, by default 6.
        n_mute : int, optional
            number of mutations, by default 3.
        """

        # initialize base class
        super().__init__(problem)

        # set parameters
        self.n_min = n_min
        self.n_max = n_max
        self.n_gen = n_gen
        self.n_cross = n_cross
        self.n_mute = n_mute

        # set default options
        self.k = 5
        self.n_crowd = 4
        self.eta = 10
        self.init_method = "lhs"
        self.mute_method = "random"
        self.multimodal = False

        # initialize population's design variables, objective, constraints, penalty and fitness
        self.x = np.zeros((self.n_max, self.problem.n_var))
        self.obj = np.zeros((self.n_max, self.problem.n_obj))
        self.const = np.zeros((self.n_max, self.problem.n_const))
        self.pen = np.zeros(self.n_max)
        self.f = np.zeros(self.n_max)
        self.f_scaled = np.zeros(self.n_max)

        # initialize matrix with distance between each individual of the population
        self.distance_matrix = np.zeros((self.n_max, self.n_max))

        # set scaling factor
        self.sf = np.linalg.norm(self.problem.bounds[:, 1] - self.problem.bounds[:, 0])

    def print(self):
        """
        Print a summary in the console.
        """

        print("################################")
        print("####### VIKGA Parameters #######")
        print("################################")
        print(f"Initial population size: {self.n_min}")
        print(f"Maximum population size: {self.n_max}")
        print(f"Maximum Generation:      {self.n_gen}")
        print("")

    def run(self):
        """
        Main function to run the algorithm.
        """

        # start optimization process
        self.start()

        # initialize the population
        self.initialize()

        # print progress
        if self.display:
            print(f"Generation {0}: {self.pop_size} individuals", end="\r")

        # loop on generations
        for self.i_gen in range(self.n_gen):

            # perform crossover
            self.crossover()

            # perform mutation
            self.mutation()

            # add individual to the population
            self.addition()

            # compute current population's fitness
            self.f[:self.pop_size] = compute_fitness(self.obj[:self.pop_size], self.pen[:self.pop_size])

            # scale current population's fitness
            if self.multimodal:
                self.scale_fitness()

            # get current generation's results
            self.results.get_gen(self)

            # termination criteria
            if self.pop_size >= self.n_max:
                if self.display:
                    print(f"\x1b[KGeneration {self.i_gen + 1}: {self.pop_size} individuals")
                    print("Maximum population size reached!")
                    print("")

                break

            # print progress
            if self.display:
                if self.i_gen < self.n_gen - 1:
                    print(f"Generation {self.i_gen + 1}: {self.pop_size} individuals", end="\r")

                else:
                    print(f"\x1b[KGeneration {self.i_gen + 1}: {self.pop_size} individuals")
                    print("Maximum generation reached!")
                    print("")

        # get final population's results
        self.results.get_pop(self)

        # stop the optimization process
        self.stop()

    def initialize(self):
        """
        Initialize the starting population.
        """

        # generate normalized sample
        x_sample = sample(self.init_method, self.n_min, self.problem.n_var)

        # adapt sample to problem's bounds
        self.x[:self.n_min] = (self.problem.bounds[:, 1]
                               - self.problem.bounds[:, 0]) * x_sample + self.problem.bounds[:, 0]

        # compute distance between each individual of the initial population
        self.distance_matrix[:self.n_min, :self.n_min] = np.linalg.norm(self.x[:self.n_min, np.newaxis]
                                                                        - self.x[:self.n_min], axis=2)

        # evaluate initial population's objective and constraints
        self.obj[:self.n_min], self.const[:self.n_min] = self.evaluate(self.x[:self.n_min])

        # compute initial population's penalty
        if self.problem.n_const > 0:
            self.pen[:self.n_min] = compute_penalty(self.const[:self.n_min], self.problem)

        # compute initial population's fitness
        self.f[:self.n_min] = compute_fitness(self.obj[:self.n_min], self.pen[:self.n_min])

        # update population size
        self.pop_size = self.n_min

        # scale initial population's fitness to optima
        if self.multimodal:
            self.scale_fitness()

        # get initial generation's results
        self.results.get_gen(self)

    def scale_fitness(self):
        """
        Scale the fitness to local optima
        """

        # identify optima
        self.id_optima = get_optima(self.x[:self.pop_size], self.f[:self.pop_size],
                                    self.distance_matrix[:self.pop_size, :self.pop_size], self.k)

        # if optima are identified
        if len(self.id_optima) > 0:
            # set array containing the scaled fitness to each optima
            f_optima = np.zeros((self.pop_size, len(self.id_optima)))

            # scale fitness to each optima
            f_optima = self.f[:self.pop_size, np.newaxis] / self.f[self.id_optima]

            # limit values to [0, 1]
            f_optima[f_optima > 1.0] = 1.0

            # adjust so that the median value of the scaled fitness is 0.5
            m = np.median(f_optima, axis=0)
            m[m >= 1.0] = 1 - 1e-10
            p = np.log(0.5) / np.log(m)
            f_optima = np.power(f_optima, p)

            # proximity-weighted scaling process of the whole fitness
            prox = 1 / (self.distance_matrix[:self.pop_size, self.id_optima] + 1e-10)
            self.f_scaled[:self.pop_size] = np.sum(f_optima * prox, axis=1) / np.sum(prox, axis=1)

        else:

            # otherwise the scaled fitness is just the normal fitness
            self.f_scaled[:self.pop_size] = self.f[:self.pop_size]

    def crossover(self):
        """
        Perform the crossover operation.
        From: M. Hall. 2012. A Cumulative Multi-Niching Genetic Algorithm for Multimodal Function Optimization.
        """

        # if multimodal problem is considered then use the scaled fitness
        if self.multimodal:
            fitness = self.f_scaled[:self.pop_size]
        else:
            fitness = self.f[:self.pop_size]

        # set array containing offsprings' design variables
        self.x_cross = np.zeros((self.n_cross, self.problem.n_var))

        # TODO: check why sometimes I have som nan in proba
        # compute fitness-proportionate probability
        p_fit = fitness / np.sum(fitness)

        # loop on crossovers
        # TODO: check if all parents could be selected at once preventing to reselect parents
        for i in range(int(self.n_cross / 2)):

            # select first parent using fitness-proportionate selection
            id_p1 = np.random.choice(np.arange(self.pop_size), p=p_fit)

            # get all population's index except the one from the first parent
            id_pop = np.setdiff1d(np.arange(self.pop_size), id_p1)

            # compute distance-proportionate probability related to the first parent
            prox = 1 / self.distance_matrix[id_p1, id_pop]
            p_dist = prox / np.sum(prox)

            # select crowd of individual around the first parent using distance-proportionate selection
            id_crowd = np.random.choice(id_pop, self.n_crowd, p=p_dist, replace=False)

            # select second parent as the fittest of the crowd
            id_p2 = id_crowd[np.where(fitness[id_crowd] == fitness[id_crowd].max())[0][0]]

            # compute offsprings' design variables
            self.x_cross[[2 * i, 2 * i + 1]] = sbx(self.x[id_p1], self.x[id_p2], self.problem.bounds, self.eta)

    def mutation(self):
        """
        Perform the mutation operation.
        """

        # TODO: implement more efficient mutation
        # set array containing offsprings' design variables
        self.x_mute = np.zeros((self.n_mute, self.problem.n_var))

        # generate random normalized sample
        x_sample = sample("random", self.n_mute, self.problem.n_var)

        # adapt sample to problem's bounds
        self.x_mute = (self.problem.bounds[:, 1] - self.problem.bounds[:, 0]) * x_sample + self.problem.bounds[:, 0]

    def addition(self):
        """
        Perform the addition operation.
        From: M. Hall. 2012. A Cumulative Multi-Niching Genetic Algorithm for Multimodal Function Optimization.
        """

        # if multimodal problem is considered then use the scaled fitness
        if self.multimodal:
            fitness = self.f_scaled[:self.pop_size]
        else:
            fitness = self.f[:self.pop_size]

        # gather all offsprings' design variables
        x_off = np.concatenate((self.x_cross, self.x_mute))

        # get old population size
        size = self.pop_size

        # reset counters
        self.c_cross = 0
        self.c_mute = 0

        # loop on offsprings
        for i in range(len(x_off)):

            # if the maximum population size is not reached try to add offspring
            if self.pop_size < self.n_max:

                # compute offspring's distance from the current population
                dist = np.linalg.norm(self.x[:self.pop_size] - x_off[i], axis=1)

                # get closest individual
                id_min = np.argmin(dist[:size])

                # compute distance threshold
                # TODO: implement an adaptive threshold
                r_min = (1 - fitness[id_min] ** (2 * 0.95)) * 0.2 + 0.001

                # if "far" enough from closest individual add to the current population
                if r_min <= dist[id_min] / self.sf:

                    # update distance matrix
                    self.distance_matrix[:self.pop_size, self.pop_size] = dist
                    self.distance_matrix[self.pop_size, :self.pop_size] = dist

                    # add offspring to the population
                    self.x[self.pop_size] = x_off[i]
                    self.pop_size += 1

                    # update counters
                    if i < self.n_cross:
                        self.c_cross += 1
                    else:
                        self.c_mute += 1

            # else break from addition operation
            else:
                break

        # check if there is a new population
        if size < self.pop_size:

            # evaluate objective and constraints of the new population
            self.obj[size:self.pop_size], self.const[size:self.pop_size] = self.evaluate(self.x[size:self.pop_size])

            # compute penalty of the new population
            if self.problem.n_const > 0:
                self.pen[size:self.pop_size] = compute_penalty(self.const[size:self.pop_size], self.problem)
