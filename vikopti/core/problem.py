import numpy as np
import matplotlib.pyplot as plt
from vikopti.core.constraint import Constraint


class Problem:
    """
    Class representing a problem.
    """

    def __init__(self, n_var: int, n_obj: int, n_const: int, plot: bool = True):
        """
        Constructs the problem object and set the different attributes.

        Parameters
        ----------
        n_var : int
            problem's number of design variables.
        n_obj : int
            problem's number of objective.
        n_const : int
            problem's number of constraints.
        plot : bool, optional
            option to tell if problem can be plotted or not, by default True.
        """

        # set parameters
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_const = n_const

        # set options
        self.plot_flag = plot

        # set constraints holder
        self.constraint = []

    def print(self):
        """
        Print a summary of the problem in the console.
        """

        print('#' * (6 + len(self.name) + 6))
        print('##### ' + self.name + ' #####')
        print('#' * (6 + len(self.name) + 6))
        print(f'N° of variables:   {self.n_var}')
        print(f'N° of objective:   {self.n_obj}')
        print(f'N° of constraints: {self.n_const}')
        print("")

    def add_constraint(self, type: str, limit: float):
        """
        Add constraint to the problem.

        Parameters
        ----------
        type : str
            constraint's type.
        limit : float
            constraint's limit.
        """

        # add constraint to the holder
        self.constraint.append(Constraint(type, limit))

    def plot_f(self, grid_size=500, fig_size=(6, 6)):
        """
        Plot the problem's objective function.

        Parameters
        ----------
        grid_size : int, optional
            grid size, by default 500.
        fig_size : tuple, optional
            figure size, by default (6,6)
        """

        # check if problem can be plotted
        if self.plot_flag:

            # if the problem is 1D
            if self.n_var == 1:

                # create figure
                fig, ax = plt.subplots(figsize=fig_size)

                # generate grid
                x = np.linspace(self.bounds[0][0], self.bounds[0][1], grid_size)

                # plot function
                ax.plot(x, self.func([x])[0], label="fitness")

                # set figure's limit to pb's bounds
                ax.set_xlim(self.bounds[0][0], self.bounds[0][1])

                # set figures's labels
                ax.set_xlabel(self.var[0])
                ax.set_ylabel('f')

                return fig, ax

            # if the problem is 2D
            elif self.n_var == 2:

                # create figure
                fig, ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": "3d"})

                # generate grid
                x = np.linspace(self.bounds[0][0], self.bounds[0][1], grid_size)
                y = np.linspace(self.bounds[1][0], self.bounds[1][1], grid_size)
                xgrid, ygrid = np.meshgrid(x, y)

                # plot function
                ax.plot_surface(xgrid, ygrid, self.func([xgrid, ygrid])[0], cmap='YlGnBu_r')

                # set figure's limit to pb's bounds
                ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
                ax.set_ylim(self.bounds[1][0], self.bounds[1][1])

                # set figures's labels
                ax.set_xlabel(self.var[0])
                ax.set_ylabel(self.var[1])
                ax.set_zlabel('f')

                return fig, ax

    def plot_f_contour(self, grid_size=500, n_contour=100, fig_size=(6, 6)):
        """
        Plot the problem's objective function as a contour plot.

        Parameters
        ----------
        grid_size : int, optional
            2 dimensional grid size, by default 100.
        n_contour : int, optional
            number of contour levels, by default 10.
        fig_size : tuple, optional
            figure size, by default (6,6)
        """

        # check if problem can be plotted
        if self.plot_flag:

            # if the problem is 2D
            if self.n_var == 2:

                # create figure
                fig, ax = plt.subplots(figsize=fig_size)

                # generate grid
                x = np.linspace(self.bounds[0][0], self.bounds[0][1], grid_size)
                y = np.linspace(self.bounds[1][0], self.bounds[1][1], grid_size)
                xgrid, ygrid = np.meshgrid(x, y)

                # plot function
                cp = ax.contourf(xgrid, ygrid, self.func([xgrid, ygrid])[0], levels=n_contour,  cmap='YlGnBu_r')

                # set figure's limit to pb's bounds
                ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
                ax.set_ylim(self.bounds[1][0], self.bounds[1][1])

                # set figures's labels
                ax.set_xlabel(self.var[0])
                ax.set_ylabel(self.var[1])

                # set colormap
                cbar = fig.colorbar(cp)
                cbar.ax.set_ylabel('f', rotation=0)

                return fig, ax
