from vikopti.core.constraint import Constraint


class Problem:
    """
    Class representing an optimization problem.
    """

    def __init__(self, n_var: int, n_obj: int, n_const: int, plotable: bool = False):
        """
        Constructs the problem object and set the different attributes.

        Parameters
        ----------
        n_var : int
            problem's number of variables.
        n_obj : int
            problem's number of objectives.
        n_const : int
            problem's number of constraints.
        plot : bool, optional
            option to tell if problem can be plotted or not, by default False.
        """

        # set parameters
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_const = n_const

        # set options
        self.plot_flag = plotable

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
        print(f'N° of objectives:   {self.n_obj}')
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
