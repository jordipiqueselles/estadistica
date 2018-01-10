from functools import partial
from functions import *
import logging


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    funService = partial(weibull, a=0.5439, b=31)
    # compare theoretical values vs real values for the Weibull distribution
    getTheoreticalValuesWeibull(0.5439, 31)
    logging.info("")
    getMetricsRndDistr(funService)
    plt.show()