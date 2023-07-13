from vikopti.problems.gl import GL
from vikopti.algorithms.vikga import VIKGA


def main():

    # define the problem
    pb = GL()

    # define the algorithm
    algo = VIKGA(pb, n_min=10, n_max=1000, n_gen=500)

    # modify some parameter
    algo.n_proc = 4
    algo.display = True
    algo.save = False

    # run the algorithm
    algo.run()


if __name__ == '__main__':
    main()
