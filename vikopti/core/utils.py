import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from vikopti.core.problem import Problem


def compute_penalty(const: np.ndarray, problem: Problem):
    """
    Compute the penalty values from constraints values.

    Parameters
    ----------
    const : np.ndarray
        constraints values.
    problem : Problem
        problem considered.

    Returns
    -------
    penalty : np.ndarray
        penalty values.
    """

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
    """
    Compute the fitness function from objectives and penalty values.

    Parameters
    ----------
    obj : np.ndarray
        objectives values.
    pen : np.ndarray
        penalty values.

    Returns
    -------
    fitness : np.ndarray
        fitness function.
    """

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
    """
    Identify optima in the population from comparing closest neighbors.
    From: M. Hall. 2012. A Cumulative Multi-Niching Genetic Algorithm for Multimodal Function Optimization.

    Parameters
    ----------
    x : np.ndarray
        population's variables.
    f : np.ndarray
        population's fitness values.
    distance : np.ndarray
        population's distance matrix.
    k : int, optional
        number of neighbors to use for comparison, by default 5.

    Returns
    -------
    id_optima : np.ndarray
        array containing the index of the optima.
    """

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
    """
    Perform the simulated binary crossover.
    From K. Deb, and all. 2007. Self-adaptive simulated binary crossover for real-parameter optimization.

    Parameters
    ----------
    xp1 : np.ndarray
        first parent's design variables.
    xp2 : np.ndarray
        second parent's design variables.
    bounds : np.ndarray
        design space boundaries.
    eta : float, optional
        distribution index, by default 2.0.

    Returns
    -------
    xo : np.ndarray
        offsprings' design variables.
    """

    # Generate random numbers for each variables
    rand = np.random.random(size=len(xp1))

    # compute beta
    beta = np.where(rand <= 0.5,
                    (2 * rand) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - rand))) ** (1 / (eta + 1)))

    # Produce offsprings
    x1 = 0.5 * ((1 + beta) * xp1 + (1 - beta) * xp2)
    x2 = 0.5 * ((1 - beta) * xp1 + (1 + beta) * xp2)

    # make sure it is in bound
    x1 = np.clip(x1, bounds[:, 0], bounds[:, 1])
    x2 = np.clip(x2, bounds[:, 0], bounds[:, 1])

    # make an array
    xo = np.array([x1, x2])

    # This for updating the distribution index but not sure yet
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

    return xo


def write_df(df, directory):
    """
    Write a panda dataframe in a nice way

    Parameters
    ----------
    df : pd.DataFrame
        dataframe considered.
    directory : str
        directory where to write the dataframe
    """

    # Get column width for better visualization
    col_w = []
    for header in list(df.columns):
        max_w = max(len(header), len(max(df[header].to_numpy(str))))
        col_w.append(max_w + 2)

    # Generate list or formatters
    fmts = [('{:<' + str(w) + '}').format for w in col_w]

    # Write df to txt
    df.to_string(os.path.join(directory, df.name + '.txt'),
                 col_space=col_w, header=True, index=True, formatters=fmts, justify="left")


def plot_addition(problem, x_off, x_nei, radius, figsize=(6, 6), grid_size=500, n_contour=50):
    """
    Plot the addition operation.

    Parameters
    ----------
    problem : Problem
        problem considered.
    x_off : np.ndarray
        offspring's variables.
    x_nei : np.ndarray
        closest neighbor's  variables.
    radius : float
        distance threshold.
    grid_size : int, optional
        2 dimensional grid size, by default 100.
    n_contour : int, optional
        number of contour levels, by default 10.
    fig_size : tuple, optional
        figure size, by default (6,6)
    """

    # if the problem is 2D
    if problem.n_var == 2:

        # create figure
        fig, ax = problem.plot_contour(figsize, grid_size, n_contour)

        # plot offspring and closest neighbor
        plt.plot(x_off[0], x_off[1], 'bo')
        plt.plot(x_nei[0], x_nei[1], 'ko')

        # Plot the circle
        circle = plt.Circle((x_nei[0], x_nei[1]), radius, color='red', fill=False)
        plt.gca().add_patch(circle)

        # make axis equal not to be fooled
        plt.axis('equal')


def plot_population(algo, figsize=(6, 6), grid_size=500, n_contour=50):
    """
    Plot the population.

    Parameters
    ----------
    algo : Algorithm
        algorithm considered
    grid_size : int, optional
        2 dimensional grid size, by default 100.
    n_contour : int, optional
        number of contour levels, by default 10.
    fig_size : tuple, optional
        figure size, by default (6,6)
    """

    # if the problem is 2D
    if algo.problem.n_var == 2:

        # create figure
        fig, ax = algo.problem.plot_contour(figsize, grid_size, n_contour)

        # plot population
        plt.scatter(algo.x[:algo.pop_size, 0], algo.x[:algo.pop_size, 1], color='black')
