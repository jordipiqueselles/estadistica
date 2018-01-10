import random as rnd
import math
import numpy as np
import matplotlib.pyplot as plt
import logging


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

    def run(self, nClients=100000, whenToPrint=100, plot=True):
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

                logging.info("L:\t" + str(self.listL[-1].round(2)) + "\t\tLq:\t" + str(self.listLq[-1].round(2)) +
                          "\t\tW:\t" + str(self.listW[-1].round(2)) + "\t\tWq\t" + str(self.listWq[-1].round(2)))

        listOcServ = self.listL - self.listLq
        if plot:
            plt.figure()
            plt.scatter(list(range(len(listOcServ))), listOcServ)
            plt.title(self.title)

        return listOcServ, self.listL, self.listLq, self.listW, self.listWq


def getMetricsRndDistr(rndDistr, nSamples=10000, plot=True):
    samples = [rndDistr() for _ in range(nSamples)]
    mean = np.mean(samples)
    variance = np.var(samples)
    coefVar = variance**0.5 / mean

    logging.info("mean real data: " + str(mean.round(2)))
    logging.info("variance real data: " + str(variance.round(2)))
    logging.info("coef variation real data: " + str(coefVar.round(2)))

    if plot:
        plt.figure()
        plt.hist(samples, bins=min(nSamples/5, 200))

    return mean, variance, coefVar


def getTheoreticalValuesWeibull(a, b):
    mean = b*math.gamma((a+1)/a)
    variance = b*b*(math.gamma((a+2)/a) - math.gamma((a+1)/a)**2)
    coefVar = variance**0.5 / mean

    logging.info("mean theoretical: " + str(round(mean, 2)))
    logging.info("variance theoretical: " + str(round(variance, 2)))
    logging.info("coef variation theoretical: " + str(round(coefVar, 2)))

    return mean, variance, coefVar
