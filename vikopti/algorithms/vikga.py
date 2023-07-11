import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from pysampling.sample import sample
from vikopti.core.problem import Problem
from vikopti.core.algorithm import Algorithm


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
        self.optima_method = "closest"

        # initialize population's design variables, objective, constraints, penalty and fitness
        self.x = np.zeros((self.n_max, self.problem.n_var))
        self.obj = np.zeros((self.n_max, self.problem.n_obj))
        self.const = np.zeros((self.n_max, self.problem.n_const))
        self.p = np.zeros(self.n_max)
        self.f = np.zeros(self.n_max)
        self.f_scaled = np.zeros(self.n_max)

        # initialize matrix with distance between each individual of the population
        self.distance_matrix = np.zeros((self.n_max, self.n_max))

        # set counters
        self.c_cross = 0
        self.c_mute = 0

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

        # initialize population
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
            self.compute_fitness()

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

                # break
                break

            # print progress
            if self.display:
                if self.i_gen < self.n_gen - 1:
                    print(f"Generation {self.i_gen + 1}: {self.pop_size} individuals", end="\r")

                else:
                    print(f"\x1b[KGeneration {self.i_gen + 1}: {self.pop_size} individuals")
                    print("Maximum generation reached!")
                    print("")

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

        # update population size
        self.pop_size = self.n_min

        # compute distance between each individual of the initial population
        self.distance_matrix[:self.pop_size, :self.pop_size] = np.linalg.norm(self.x[:self.pop_size, np.newaxis]
                                                                              - self.x[:self.pop_size], axis=2)

        # evaluate initial population's objective and constraints
        self.obj[:self.pop_size], self.const[:self.pop_size] = self.evaluate(self.x[:self.pop_size])

        # get initial population's results
        self.results.get_pop(self.x[:self.pop_size], self.obj[:self.pop_size], self.const[:self.pop_size])

        # compute initial population's penalty
        if self.problem.n_const > 0:
            self.p[:self.pop_size] = self.compute_penalty(self.const[:self.pop_size])

        # compute initial population's fitness
        self.compute_fitness()

        # scale initial population's fitness
        if self.multimodal:
            self.scale_fitness()

        # get initial generation's results
        self.results.get_gen(self)

    def scale_fitness(self, eps=1e-10):
        """
        Scale the fitness of the current population.

        Parameters
        ----------
        eps : float, optional
            small value to prevent zero division, by default 1e-10
        """

        # identify optima in the current population.
        self.get_optima()

        # if optima are identified
        if len(self.id_optima) > 0:

            # initialize scaled fitness to optima
            f_optima = np.zeros((self.pop_size, len(self.id_optima)))

            # scale fitness to each optima
            f_max = self.f[self.id_optima]
            f_optima = self.f[:self.pop_size, np.newaxis] / f_max

            # limit values to [0, 1]
            f_optima[f_optima > 1.0] = 1.0

            # adjust that the median value is 0.5
            m = np.median(f_optima, axis=0)
            m[m >= 1.0] = 1 - eps
            p = np.log(0.5) / np.log(m)
            f_optima = np.power(f_optima, p)

            # proximity-weighted scaling process of the whole fitness
            prox = 1 / (self.distance_matrix[:self.pop_size, self.id_optima] + eps)
            self.f_scaled[:self.pop_size] = np.sum(f_optima * prox, axis=1) / np.sum(prox, axis=1)

        else:
            self.f_scaled[:self.pop_size] = self.f[:self.pop_size]

    def get_optima(self):
        """
        Identify optima in the current population.
        """

        # switch on method
        if self.optima_method == "closest":

            # get the k closest neighbors
            id_neighbors = np.argsort(self.distance_matrix[:self.pop_size, :self.pop_size], axis=1)[:, 1:self.k + 1]

            # get where fitness is better than all its neighbors
            mask = np.all(self.f[:self.pop_size, np.newaxis] > self.f[id_neighbors], axis=1)

            # get optima index
            self.id_optima = np.where(mask)[0]

        if self.optima_method == "kdtree":

            # initialize tree
            tree = KDTree(self.x[:self.pop_size], metric="euclidean")

            # initialize list of optima's index
            optima = []

            # loop on individuals
            for i in range(self.pop_size):

                # get neighbors index excluding current individual
                dist, id = tree.query(self.x[i].reshape(1, -1), k=self.k + 1)
                id = id[0][1:]

                # if fitness better than all neighbors then it is an optimum
                if all(self.f[i] > self.f[id]):
                    optima.append(i)

            # get optima index
            self.id_optima = np.array(optima)

        if self.optima_method == "dbscan":

            # initialize list of optima's index
            optima = []

            # apply DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan.fit(self.x[:self.pop_size])
            labels = dbscan.labels_

            # get unique cluster labels
            unique_labels = np.unique(labels)

            if len(unique_labels) > 1:
                for lab in unique_labels[1:]:
                    optimum = np.arange(self.pop_size)[labels == lab][np.argmax(self.f[:self.pop_size][labels == lab])]
                    optima.append(optimum)

            # get optima index
            self.id_optima = np.array(optima)

    def plot_fitness(self, f_optima):

        if self.problem.n_var == 1:
            fig, ax = self.problem.plot_f(500)
            for i in range(len(f_optima[0])):
                ax.plot(self.x[:self.pop_size], f_optima[:self.pop_size, i], '.')

            fig, ax = self.problem.plot_f(500)
            ax.plot(self.x[:self.pop_size], self.f_scaled[:self.pop_size], '.', label="scaled fitness")

    def crossover(self):
        """
        Perform the crossover operation.
        From: M. Hall. 2012. A Cumulative Multi-Niching Genetic Algorithm for Multimodal Function Optimization.
        """

        if self.multimodal:
            fitness = self.f_scaled[:self.pop_size]
        else:
            fitness = self.f[:self.pop_size]

        # set array containing offsprings' design variables
        self.x_cross = np.zeros((self.n_cross, self.problem.n_var))

        # compute fitness-proportionate probability
        p_fit = fitness / np.sum(fitness)

        parents = []
        # loop on crossovers
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
            self.x_cross[[2 * i, 2 * i + 1]] = self.sbx(self.x[[id_p1, id_p2]], self.eta)

            parents.append(id_p1)
            parents.append(id_p2)

    def mutation(self):
        """
        Perform the mutation operation.
        """

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
                r_min = (1 - fitness[id_min] ** (2 * 0.95)) * 0.1

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

            # get new population's results
            self.results.get_pop(self.x[size:self.pop_size],
                                 self.obj[size:self.pop_size], self.const[size:self.pop_size])

            # compute penalty of the new population
            if self.problem.n_const > 0:
                self.p[size:self.pop_size] = self.compute_penalty(self.const[size:self.pop_size])
