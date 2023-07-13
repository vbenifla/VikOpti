import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vikopti.core.utils import write_df


class Results:
    """
    Class representing results.
    """

    def __init__(self, problem, save):
        """
        Construct the results object and set the different attributes.

        Parameters
        ----------
        problem : Problem
            problem considered.
        save : bool, optional
            option to save results, by default True.
        """

        # set parameters
        self.problem = problem

        # make results directory
        if save:
            self.make_directory()

        # set results dataframe
        self.df_gen = pd.DataFrame(columns=["size", "crossovers", "mutations", "optimum"]
                                   + self.problem.var
                                   + [f'obj{i+1}' for i in range(self.problem.n_obj)]
                                   + [f'const{i+1}' for i in range(self.problem.n_const)])

    def make_directory(self):
        """
        Make the directory where results are saved.
        """

        # set results directory
        base_dir = os.getcwd()
        dir_name = time.strftime("%Y_%m_%d_%Hh%M")
        self.dir = os.path.join(base_dir, "results", dir_name)

        # make results directory
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)

    def get_gen(self, algo):
        """
        Get current generation's results.

        Parameters
        ----------
        algo : Algorithm
            algorithm being run.
        """

        # get algorithm's population size and number of offsprings added by crossover and mutation
        data = [algo.pop_size, algo.c_cross, algo.c_mute]

        # get the optimum of the algorithm's population
        i = np.argmax(algo.f[:algo.pop_size])

        # get optimum's results
        data += [i]
        data += [algo.x[i, j] for j in range(self.problem.n_var)]
        data += [algo.obj[i, j] for j in range(self.problem.n_obj)]
        data += [algo.const[i, j] for j in range(self.problem.n_const)]

        # set dataframe
        self.df_gen.loc[len(self.df_gen.index)] = data

    def get_pop(self, algo):
        """
        Get current population's results.

        Parameters
        ----------
        algo : Algorithm
            algorithm being run.
        """

        # Create the DataFrame
        self.df_pop = pd.DataFrame(np.hstack((algo.x[:algo.pop_size],
                                              algo.obj[:algo.pop_size],
                                              algo.const[:algo.pop_size])),

                                   columns=self.problem.var
                                   + [f'obj{i+1}' for i in range(self.problem.n_obj)]
                                   + [f'const{i+1}' for i in range(self.problem.n_const)])

    def save(self):
        """
        Save the results.
        """

        # round to decimals for better visualization
        self.df_gen = self.df_gen.round(5)
        self.df_pop = self.df_pop.round(5)

        # set name for saving
        self.df_gen.name = "generation"
        self.df_pop.name = "population"

        # loop on dataframes
        for df in [self.df_gen, self.df_pop]:
            write_df(df, self.dir)

    def print(self):
        """
        Print a summary of the results in the console.
        """

        print("###################")
        print("##### Results #####")
        print("###################")
        print(f"Run time (s): {self.run_time}")
        print(f"N° evaluations: {self.c_eval}")
        print("Optimum:")
        print('\n'.join(str(self.df_gen.iloc[-1][4:]).split('\n')[:-1]))
        print("")
        self.plot()

    def plot(self):
        """
        Plot the results.
        """

        self.plot_distribution()
        self.plot_objective()
        self.plot_constraint()
        plt.show()

    def plot_distribution(self):
        """
        Plot the variables distribution.
        """

        # create an instance of the PairGrid class
        grid = sns.PairGrid(data=self.df_pop, vars=self.problem.var)

        # map a histogram to the diagonal
        # grid = grid.map_diag(sns.kdeplot, color = 'darkred')
        grid.map_diag(plt.hist, bins=10, color='darkred', edgecolor='k')

        # map a density plot to the lower triangle
        grid.map_upper(plt.scatter, color='darkred', s=1)

        # map a scatter plot to the upper triangle
        grid.map_lower(sns.kdeplot, clip=self.problem.bounds, cmap='Reds')

    def plot_objective(self, figsize=(6, 6)):
        """
        Plot the objective evolution.
        """

        # create figure
        fig, ax = plt.subplots(nrows=self.problem.n_obj, ncols=1, figsize=figsize)

        # plot objective
        self.df_gen['obj1'].plot(ax=ax)

        # set figures's labels
        ax.set_ylabel('objective')
        ax.set_xlabel('Generation')

    def plot_constraint(self, figsize=(6, 6)):
        """
        Plot the constraints evolution.
        """

        if self.problem.n_const > 0:

            # create figure
            fig, ax = plt.subplots(nrows=self.problem.n_const, ncols=1, figsize=figsize)

            # make share axis in abscise
            ax = ax if self.problem.n_const > 1 else [ax]
            for a in ax:
                ax[0].get_shared_x_axes().join(ax[0], a)

            # create different color for each constraint
            color = iter(plt.cm.rainbow(np.linspace(0, 1, self.problem.n_const)))

            # loop on constraints
            for i in range(self.problem.n_const):
                c = next(color)
                self.df_gen["const" + str(i+1)].plot(ax=ax[i], c=c, style='-')
                ax[i].axhline(y=self.problem.constraint[i].limit, c=c, linestyle=':')
                ax[i].set_ylabel("const" + str(i))
                ax[i].set_xlabel('Generation')
