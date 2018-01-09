import random as rnd
import itertools as ittls
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.stats as st

# Weibull
def weibull(a, b):
    y = rnd.random()
    x = b*(-math.log(1-y))**(1/a)
    return x

class Gg1Simulation:
    def __init__(self, title, arrivalTimeDistr, serviceTimeDistr):
        self.title = title
        self.arrivalTimeDistr = arrivalTimeDistr
        self.serviceTimeDistr = serviceTimeDistr

    def reset(self, sizeArr):
        self.L = 0  # total occupancy
        self.Lq = 0  # occupancy of the queue
        self.W = 0  # total waiting time
        self.Wq = 0  # waiting time in the queue

        # lists to save the statistics at some points
        self.listL = np.zeros((sizeArr, 1))
        self.listLq = np.zeros((sizeArr, 1))
        self.listW = np.zeros((sizeArr, 1))
        self.listWq = np.zeros((sizeArr, 1))

    def run(self, nClients=100000, whenToPrint=100, prnt=True, plot=True):
        self.reset(nClients//whenToPrint - 1)
        t_last = 0  # arrival time to the system for the last client
        ohm_last = 0  # exit time for the last client
        for i in range(nClients):
            ti = t_last + self.arrivalTimeDistr()  # arrival time to the system
            tis = max(ti, ohm_last)  # arrival time to the server
            wqi = tis - ti  # waiting time in the queue
            serviceTime = self.serviceTimeDistr()  # service time
            wi = wqi + serviceTime  # time in the system
            t_last = ti  # update t_last
            ohm_last = wi + ti  # update ohm_last

            # update statistics
            self.L += wi
            self.Lq += wqi
            self.W += wi
            self.Wq += wqi

            if i != 0 and i%whenToPrint == 0:
                # update
                idx = i // whenToPrint - 1
                self.listL[idx] = self.L/ti
                self.listLq[idx] = self.Lq/ti
                self.listW[idx] = self.W/(i+1)
                self.listWq[idx] = self.Wq/(i+1)

                if prnt:
                    print("L:\t", round(self.listL[-1], 2), "\t\tLq:\t", round(self.listLq[-1], 2),
                          "\t\tW:\t", round(self.listW[-1], 2), "\t\tWq\t", round(self.listWq[-1], 2))

        listOcServ = self.listL - self.listLq
        if plot:
            plt.figure()
            plt.scatter(list(range(len(listOcServ))), listOcServ)
            plt.title(self.title)

        return listOcServ, self.listL, self.listLq, self.listW, self.listWq


def getMetricsRndDistr(rndDistr, nSamples=10000, verbose=True):
    samples = [rndDistr() for _ in range(nSamples)]
    mean = np.mean(samples)
    variance = np.var(samples)
    coefVar = variance**0.5 / mean
    if verbose:
        print("mean sample:", mean)
        print("variance sample:", variance)
        print("coef variation:", coefVar)
        plt.figure()
        plt.hist(samples, bins=min(nSamples/5, 100))
    return mean, variance, coefVar


def getTheoreticalValuesWeibull(a, b, verbose=True):
    mean = b*math.gamma((a+1)/a)
    variance = b*b*(math.gamma((a+2)/a) - math.gamma((a+1)/a)**2)
    coefVar = variance**0.5 / mean
    if verbose:
        print("mean theoretical:", mean)
        print("variance theoretical:", variance)
        print("coef variation theoretical:", coefVar)
    return mean, variance, coefVar


funArrivals = partial(weibull, a=2, b=88)
funService = partial(weibull, a=0.5439, b=40)

# compare theoretical values vs real values for the Weibull distribution
getTheoreticalValuesWeibull(2, 88)
print()
getMetricsRndDistr(funArrivals)

listP = [0.4, 0.7, 0.85, 0.925]
nRep = 5
for p in listP:
    print("\n\np:", p)

    bService = p * (88 * math.gamma((2 + 1) / 2)) / math.gamma((0.5439 + 1) / 0.5439)
    funService = partial(weibull, a=0.5439, b=bService)
    Lq = []
    Wq = []

    for i in range(nRep):
        print("Repetition ", (i+1))

        rnd.seed(p*1000+i)
        simulator = Gg1Simulation('Simulation rho ' + str(p), funArrivals, funService)
        (listOcServ, listL, listLq, listW, listWq) = simulator.run(100000, 100, prnt=False, plot=False)

        Lq.append(listLq[-1])
        Wq.append(listWq[-1])
        print("Mean OcServ:", listOcServ[-1].round(3))
        print("Mean L:", listL[-1].round(2))
        print("Mean Lq:", listLq[-1].round(2))
        print("Mean W:", listW[-1].round(2))
        print("Mean Wq:", listWq[-1].round(2))
        print()

    (_, _, Cs) = getTheoreticalValuesWeibull(0.5439, bService, True)
    (_, _, Ca) = getTheoreticalValuesWeibull(2, 88, False)
    allenCuneen = (p*p/(1-p)) * ((Cs**2 + Ca**2)/2)
    print("Allen-Cuneen:", round(allenCuneen, 2))
    confIntLq = st.t.interval(0.95, nRep-1, loc=np.mean(Lq), scale=st.sem(Lq))
    print("Confidence interval for Lq:", confIntLq)
    confIntWq = st.t.interval(0.95, nRep-1, loc=np.mean(Lq), scale=st.sem(Lq))
    print("Confidence interval for Wq:", confIntWq)

plt.show()
