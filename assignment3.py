import random as rnd
import itertools as ittls
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

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

    def reset(self):
        self.L = 0 # total occupancy
        self.Lq = 0 # occupancy of the queue
        self.W = 0 # total waiting time
        self.Wq = 0 # waiting time in the queue

        # lists to save the statistics at some points
        self.listL = []
        self.listLq = []
        self.listW = []
        self.listWq = []

    def run(self, nClients=100000, whenToPrint=100, prnt=True, plot=True):
        self.reset()
        t_last = 0 # arrival time to the system for the last client
        ohm_last = 0 # exit time for the last client
        for i in range(nClients):
            ti = t_last + self.arrivalTimeDistr() # arrival time to the system
            tis = max(ti, ohm_last) # arrival time to the server
            wqi = tis - ti # waiting time in the queue
            wi = self.serviceTimeDistr() # service time
            t_last = ti # update t_last
            ohm_last = tis + wi # exit time

            # update statistics
            self.L += wi
            self.Lq += wqi
            self.W += wi
            self.Wq += wqi

            if i != 0 and i%whenToPrint == 0:
                # update
                self.listL.append(self.L/ti)
                self.listLq.append(self.Lq/ti)
                self.listW.append(self.W/(i+1))
                self.listWq.append(self.Wq/(i+1))

                if prnt:
                    print("L:\t", round(self.listL[-1], 2), "\t\tLq:\t", round(self.listLq[-1], 2),
                          "\t\tW:\t", round(self.listW[-1], 2), "\t\tWq\t", round(self.listWq[-1], 2))
        if plot:
            plt.figure()
            plt.scatter(list(range(len(self.listL))), self.listL)
            plt.title(self.title)

        return (self.listL, self.listLq, self.listW, self.listWq)


def getMetricsRndDistr(rndDistr, nSamples=10000):
    samples = [rndDistr() for _ in range(nSamples)]
    mean = np.mean(samples)
    variance = np.var(samples)
    coefVar = variance**0.5 / mean
    print("mean sample:", mean)
    print("variance sample:", variance)
    print("coef variation:", coefVar)
    plt.figure()
    plt.hist(samples, bins=min(nSamples/5, 100))
    return(mean, variance, coefVar)


def getTheoreticalValuesWeibull(a, b):
    mean = b*math.gamma((a+1)/a)
    variance = b*b*(math.gamma((a+2)/a) - math.gamma((a+1)/a)**2)
    coefVar = variance**0.5 / mean
    print("mean theoretical:", mean)
    print("variance theoretical:", variance)
    print("coef variation theoretical:", coefVar)
    return(mean, variance, coefVar)


funArrivals = partial(weibull, a=2, b=88)
funService = partial(weibull, a=0.5439, b=40)

# compare theoretical values vs real values for the Weibull distribution
getTheoreticalValuesWeibull(2, 88)
getMetricsRndDistr(funArrivals)

listP = [0.4, 0.7, 0.85, 0.925]
nRep = 10
for p in listP:
    print("\n\np:", p)
    for i in range(nRep):
        print("Repetition ", (i+1))

        bService = p * (88*math.gamma((2+1)/2)) / math.gamma((0.5439+1)/0.5439)
        funService = partial(weibull, a=0.5439, b=bService)
        simulator = Gg1Simulation('Simulation rho ' + str(p), funArrivals, funService)
        (listL, listLq, listW, listWq) = simulator.run(100000, 100, prnt=False, plot=False)

        print("Mean L:", round(listL[-1], 3))
        print("Mean Lq:", round(listLq[-1], 2))
        print("Mean W:", round(listW[-1], 2))
        print("Mean Wq:", round(listWq[-1], 2))
        print()
plt.show()
