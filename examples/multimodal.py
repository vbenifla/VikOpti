from vikopti.problems.thc import THC
from vikopti.problems.f4 import F4
from vikopti.problems.f2 import F2
from vikopti.algorithms.vikga import VIKGA


def main():

    # define the problem
    for pb in [F2(), THC(), F4()]:
        pb.plot()

    # define the algorithm
    algo = VIKGA(pb, n_min=10, n_max=100, n_gen=500)

    # modify some parameter
    algo.n_cross = 4
    algo.n_mute = 2
    algo.n_proc = 4
    algo.k = 4
    algo.multimodal = True
    algo.display = True
    algo.save = False

    # run the algorithm
    algo.run()


if __name__ == '__main__':
    main()
