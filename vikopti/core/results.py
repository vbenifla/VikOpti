import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from vikopti.core.problem import Problem


class Results:
    """
    Class representing results.
    """

    def __init__(self, problem: Problem, save: bool = True):
        """
        Construct the results object and set the different attributes.

        Parameters
        ----------
        problem : Problem
            problem considered.
        save : bool, optional
            option to save results, by default True.
        """

        # make results directory
        if save:
            self.make_directory()

        # set problem
        self.problem = problem

        # set results dataframe
        self.df_gen = pd.DataFrame(columns=["size", "crossovers", "mutations", "optimum"] + self.problem.var
                                   + ["obj"] + ["const" + str(i) for i in range(self.problem.n_const)])
        self.df_pop = pd.DataFrame(columns=self.problem.var
                                   + ["obj"] + ["const" + str(i) for i in range(self.problem.n_const)])

    def make_directory(self, base_dir=os.getcwd()):
        """
        Make the directory where results are saved.

        Parameters
        ----------
        base_dir : str, optional
            directory where results are saved, by default os.getcwd().
        """

        # set results directory
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
        data += [i] + [algo.x[i, j] for j in range(self.problem.n_var)]
        + [algo.obj[i, 0]] + [algo.const[i, j] for j in range(self.problem.n_const)]

        # set dataframe
        self.df_gen.loc[len(self.df_gen.index)] = data

    def get_pop(self, x, obj, const):
        """
        Get current population's results.

        Parameters
        ----------
        algo : Algorithm
            algorithm being run.
        """

        # get algorithm's population
        for i in range(len(x)):
            data = []

            # get the design variables
            for j in range(self.problem.n_var):
                data += [x[i, j]]

            # get the objective values
            for j in range(self.problem.n_obj):
                data += [obj[i, j]]

            # get the constraints values
            for j in range(self.problem.n_const):
                data += [const[i, j]]

            # set dataframe
            self.df_pop.loc[len(self.df_pop.index)] = data

    def print(self):
        """
        Print a summary in the console.
        """

        # TODO: update summary
        print("###################")
        print("##### Results #####")
        print("###################")
        print(f"Run time (s): {self.run_time}")
        print(f"N° evaluations: {self.f_eval}")
        print("Optimum:")
        print(self.df_gen[self.problem.var + ["obj"]
                          + ["const" + str(i) for i in range(self.problem.n_const)]].iloc[-1].to_string())
        print("")

    def plot(self):
        """
        Plot all results.
        """

        self.plot_distribution()
        self.plot_objective()
        self.plot_constraint()

    def plot_distribution(self):
        """
        Plot the design variables distribution
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

    def plot_objective(self):
        """
        Plot the objective evolution.
        """

        # create figure
        fig, ax = plt.subplots(figsize=(6, 6))

        # plot objective
        self.df_gen['obj'].plot(ax=ax)

        # set figures's labels
        ax.set_ylabel('objective')
        ax.set_xlabel('Generation')

    def plot_constraint(self):
        """
        Plot the constraints evolution.
        """

        if self.problem.n_const > 0:

            # create figure
            fig, ax = plt.subplots(nrows=self.problem.n_const, ncols=1, figsize=(6, 6))

            # make share axis in abscise
            ax = ax if self.problem.n_const > 1 else [ax]
            for a in ax:
                ax[0].get_shared_x_axes().join(ax[0], a)

            # create different color for each constraint
            color = iter(cm.rainbow(np.linspace(0, 1, self.problem.n_const)))

            # loop on constraints
            for i in range(self.problem.n_const):
                c = next(color)
                self.df_gen["const" + str(i)].plot(ax=ax[i], c=c, style='-')
                ax[i].axhline(y=self.problem.constraint[i].limit, c=c, linestyle=':')
                ax[i].set_ylabel("const" + str(i))
                ax[i].set_xlabel('Generation')

    def save(self, decimals=5):
        """
        Save the results.
        """

        # round to decimals for better visualization
        self.df_gen = self.df_gen.round(decimals)
        self.df_pop = self.df_pop.round(decimals)

        # set name for saving
        self.df_gen.name = "generation"
        self.df_pop.name = "population"

        # loop on dataframes
        for df in [self.df_gen, self.df_pop]:

            # Get column width for better visualization
            col_w = []
            for header in list(df.columns):
                max_w = max(len(header), len(max(df[header].to_numpy(str))))
                col_w.append(max_w + 2)

            # Generate list or formatters
            fmts = [('{:<' + str(w) + '}').format for w in col_w]

            # Write df to txt
            df.to_string(os.path.join(self.dir, df.name + '.txt'),
                         col_space=col_w, header=True,
                         index=True, formatters=fmts, justify="left")
