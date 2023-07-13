import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from vikopti.core.problem import Problem


def compute_penalty(const: np.ndarray, problem: Problem):

    # set array containing penalty
    penalty = np.zeros(len(const))

    # loop on constraints
    for i in range(problem.n_const):

        # compute absolute constraint violation
        v = np.abs(problem.constraint[i].limit - const[:, i])

        # apply penalty only where the constraint is not respected
        if problem.constraint[i].type == "inf":
            penalty += np.where(const[:, i] > problem.constraint[i].limit, v, 0)

        elif problem.constraint[i].type == "sup":
            penalty += np.where(const[:, i] < problem.constraint[i].limit, v, 0)

    return penalty


def compute_fitness(obj: np.ndarray, pen: np.ndarray):

    # set array containing penalty
    fitness = np.zeros(len(obj))

    # find feasible solution
    feasible = pen == 0

    # find worst feasible solution get its objective
    if len(feasible) != 0:
        f_worst = np.min(pen[feasible])
    else:
        f_worst = 0

    # compute fitness
    fitness[feasible] = obj[:, 0][feasible]
    fitness[~feasible] = f_worst - pen[~feasible]

    # scale fitness between 0 and 1
    fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min())

    return fitness


def get_optima(x: np.ndarray, f: np.ndarray, distance: np.ndarray, k=5):

    # get the index of the k closest neighbors for each individual
    id_neighbors = np.argsort(distance, axis=1)[:, 1:k + 1]

    # find out for each individual if fitness is better than all its neighbors
    mask = np.all(f[:, np.newaxis] > f[id_neighbors], axis=1)

    # get each optima index
    id_optima = np.where(mask)[0]

    return id_optima


def get_optima_kdtree(x: np.ndarray, f: np.ndarray, k=5):

    # get size of the population
    pop_size = len(x)

    # initialize tree
    tree = KDTree(x, metric="euclidean")

    # initialize list of optima's index
    optima = []

    # loop on individuals
    for i in range(pop_size):

        # get neighbors index excluding current individual
        dist, id = tree.query(x[i].reshape(1, -1), k=k + 1)
        id = id[0][1:]

        # if fitness better than all neighbors then it is an optimum
        if all(f[i] > f[id]):
            optima.append(i)

    # get optima index
    id_optima = np.array(optima)

    return id_optima


def get_optima_dbscan(x: np.ndarray, f: np.ndarray, eps=0.5, min_samples=5):

    # get size of the population
    pop_size = len(x)

    # initialize list of optima's index
    optima = []

    # apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(x)
    labels = dbscan.labels_

    # get unique cluster labels
    unique_labels = np.unique(labels)

    if len(unique_labels) > 1:
        for lab in unique_labels[1:]:
            optimum = np.arange(pop_size)[labels == lab][np.argmax(f[labels == lab])]
            optima.append(optimum)

    # get optima index
    id_optima = np.array(optima)

    return id_optima


def sbx(xp1, xp2, bounds, eta=2.0):

    # Generate random numbers for each variables
    rand = np.random.random(size=len(xp1))

    # compute beta
    beta = np.where(rand <= 0.5,
                    (2 * rand) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - rand))) ** (1 / (eta + 1)))

    # Produce offsprings
    y1 = 0.5 * ((1 + beta) * xp1 + (1 - beta) * xp2)
    y2 = 0.5 * ((1 - beta) * xp1 + (1 + beta) * xp2)

    # make sure it is in bound
    y1 = np.clip(y1, bounds[:, 0], bounds[:, 1])
    y2 = np.clip(y2, bounds[:, 0], bounds[:, 1])

    # This for updating the distribution index buuuuut not sure yet
    # # Update eta based on child performance
    # child_better = False  # Flag to check if child is better than parents
    # child_worse = False  # Flag to check if child is worse than parents

    # # Check if the child is better or worse than both parents
    # if np.any(y1 != xp1) and np.any(y1 != xp2):
    #     if f(y1) < f(xp1) and f(y1) < f(xp2):
    #         child_better = True
    #     elif f(y1) > f(xp1) and f(y1) > f(xp2):
    #         child_worse = True

    # if np.any(y2 != xp1) and np.any(y2 != xp2):
    #     if f(y2) < f(xp1) and f(y2) < f(xp2):
    #         child_better = True
    #     elif f(y2) > f(xp1) and f(y2) > f(xp2):
    #         child_worse = True

    # # Update eta based on child performance
    # if child_better:
    #     eta *= 0.95  # Decrease eta by 5%
    # elif child_worse:
    #     eta *= 1.05  # Increase eta by 5%

    return np.array([y1, y2])
