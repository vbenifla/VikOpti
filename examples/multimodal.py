from vikopti.problems.f4 import F4
from vikopti.algorithms.vikga import VIKGA


def main():

    # define the problem
    pb = F4()

    # define the algorithm
    algo = VIKGA(pb, n_min=10, n_max=1000, n_gen=500)

    # modify some parameter
    algo.multimodal = True
    algo.k = 5
    algo.n_proc = 4
    algo.display = True
    algo.save = False

    # run the algorithm
    algo.run()


if __name__ == '__main__':
    main()
