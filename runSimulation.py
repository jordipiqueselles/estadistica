from functools import partial
import scipy.stats as st
import sys
from functions import *


def usage(exitCode):
    print("USAGE:")
    print("python3 runSimulation.py numberUsers numberRepetitions [listLoadingFactor] [-v] [-p]")
    print()
    print("python3 runSimulation.py 100000 10")
    print("python3 runSimulation.py 100000 10 [0.4, 0.7, 0.85, 0.925]")
    print()
    print("-v -> verbose")
    print("-p -> plot")
    exit(exitCode)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Not enough arguments \n")
        usage(1)

    if '-h' in sys.argv:
        usage(0)

    if '-v' in sys.argv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if '-p' in sys.argv:
        plot = True
    else:
        plot = False

    try:
        nUsers = int(sys.argv[1])
    except:
        print("Invalid argument for number of users", sys.argv[3])
        exit(1)

    try:
        nRep = int(sys.argv[2])
    except:
        print("Invalid argument for number of repetitions", sys.argv[3])
        exit(1)

    try:
        listP = eval(sys.argv[3])
    except:
        listP = None
    if type(listP) is not list or all((type(elem) in [float, int] and 0 <= elem for elem in listP)):
        logging.info("Taking default loadingFactors")
        listP = [0.4, 0.7, 0.85, 0.925]

    for p in listP:
        print("\n\np:", p)
        funArrivals = partial(weibull, a=2, b=88)
        bService = p * (88 * math.gamma((2 + 1) / 2)) / math.gamma((0.5439 + 1) / 0.5439)
        funService = partial(weibull, a=0.5439, b=bService)
        Lq = []
        Wq = []

        for i in range(nRep):
            print("Repetition ", (i+1))

            rnd.seed(p*1000+i)
            simulator = Gg1Simulation('Simulation rho ' + str(p), funArrivals, funService)
            (listOcServ, listL, listLq, listW, listWq) = simulator.run(nUsers, 100, plot)

            Lq.append(listLq[-1])
            Wq.append(listWq[-1])
            print("Mean OcServ:", listOcServ[-1].round(3))
            print("Mean L:", listL[-1].round(2))
            print("Mean Lq:", listLq[-1].round(2))
            print("Mean W:", listW[-1].round(2))
            print("Mean Wq:", listWq[-1].round(2))
            print()

        (_, _, Cs) = getTheoreticalValuesWeibull(0.5439, bService)
        (_, _, Ca) = getTheoreticalValuesWeibull(2, 88)
        allenCuneen = (p*p/(1-p)) * ((Cs**2 + Ca**2)/2)
        print("Allen-Cuneen:", round(allenCuneen, 2))
        confIntLq = [round(elem[0], 2) for elem in st.t.interval(0.95, nRep-1, loc=np.mean(Lq), scale=st.sem(Lq))]
        print("Confidence interval for Lq:", confIntLq)
        confIntWq = [round(elem[0], 2) for elem in st.t.interval(0.95, nRep-1, loc=np.mean(Wq), scale=st.sem(Wq))]
        print("Confidence interval for Wq:", confIntWq)

    plt.show()
