import os
import time
import numpy as np
import pandas as pd
from vikopti.core.utils import writte_df


class Results:
    """
    Class representing results.
    """

    def __init__(self, algo):
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
        if algo.save:
            self.make_directory()

        # set results dataframe
        self.df_gen = pd.DataFrame(columns=["size", "crossovers", "mutations", "optimum"]
                                   + algo.problem.var
                                   + [f'obj{i+1}' for i in range(algo.problem.n_obj)]
                                   + [f'const{i+1}' for i in range(algo.problem.n_const)])

    def make_directory(self):
        """
        Make the directory where results are saved.

        Parameters
        ----------
        base_dir : str, optional
            directory where results are saved, by default os.getcwd().
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
        data += [algo.x[i, j] for j in range(algo.x.shape[1])]
        data += [algo.obj[i, j] for j in range(algo.obj.shape[1])]
        data += [algo.const[i, j] for j in range(algo.const.shape[1])]

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

                                   columns=algo.problem.var
                                   + [f'obj{i+1}' for i in range(algo.problem.n_obj)]
                                   + [f'const{i+1}' for i in range(algo.problem.n_const)])

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
            writte_df(df, self.dir)

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
        print('\n'.join(str(self.df_gen.iloc[-1][4:]).split('\n')[:-1]))
        print("")
