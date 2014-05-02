from __future__ import division
import random
import math
import numpy as np
from matplotlib.pyplot import *
import example2sys as e2s
import pod
import time

def runAnalysis(n, k, N=1, example='butter', T=20, sigma=1., integrator='dopri5'):
    sys = e2s.example2sys(example + '_' + str(n) + '.mat')
    rsys = sys.truncate(k)

    results = []
    reducedResults = []
    controls = []

    for i in range(N):
        Y, Yhat, U = randomRuns(sys, rsys, T, sigma, integrator=integrator)
        results.append(Y)
        reducedResults.append(Yhat)
        controls.append(U)

        error = [math.fabs(y-yhat) for y, yhat in zip(Y, Yhat)]
        subplot(3, 1, 1)
        plotResults(error, T, label='error '+str(i+1))
        subplot(3, 1, 2)
        plotResults(Y, T, label='Y '+str(i+1))
        plotResults(Yhat, T, label='Yhat '+str(i+1))
        subplot(3, 1, 3)
        plotResults(U, T, label='U '+str(i+1))

    #return Y, Yhat, T, U

def x0(n):
    return np.array([0. for _n in range(n)])

def plotResults(Y, T, label=None, legend_loc='upper left', show_legend=False):
    timeSteps = range(1, T+1)
    plot(timeSteps, Y, label=label)
    if show_legend:
        legend(loc=legend_loc)
    
def randomRuns(sys, rsys, T, sigma=10.0, integrator='dopri5'):
    timeSteps = range(1, T+1)
    U = [np.array([random.gauss(0.,sigma)]) for t in timeSteps]

    sys.setupODE(integrator=integrator)
    rsys.setupODE(integrator=integrator)

    print('System of order {}'.format(sys.order))
    with Timer():
        Y = sys(timeSteps, U)
    print('System of order {}'.format(rsys.order))
    with Timer():
        Yhat = rsys(timeSteps, U)

    return Y, Yhat, U

class Timer(object):
    """Allows some basic profiling"""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, ty, val, tb):
        end = time.time()
        self.elapsed = end - self.start
        print('Time elapsed {} seconds'.format(self.elapsed))
        return False

        
