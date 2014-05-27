from __future__ import division
import random
import math
import numpy as np
from scipy import linalg
from matplotlib.pyplot import plot, subplot, legend
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

def optionPricingComparison(N=1000, k=None,
                            option="put", r=0.05, T=1., K=100., L=None):
    if k is None:
        k = max(1,int(N/50))

    print "SETUP\n===================="
    print "original system"
    with Timer():
        sys = e2s.optionPricing(N=N, option=option, r=r, T=T, K=K, L=L)

    print "auto truncated"
    with Timer():
        sys_auto_truncated = \
            pod.lss(sys, reduction="truncation_square_root_trans_matrix")
        sys_auto_truncated.x0 = np.dot(sys_auto_truncated.Ti, sys.x0)

    print "balanced truncated with k =", k
    with Timer():
        sys_balanced_truncated = \
            pod.lss(sys, reduction="truncation_square_root_trans_matrix", k=k)
        sys_balanced_truncated.x0 = np.dot(sys_balanced_truncated.Ti, sys.x0)

    print "controllability gramian reduction"
    with Timer(): 
        sys_control_truncated = \
            pod.lss(sys, reduction="controllability_truncation", k=k)
        sys_control_truncated.x0 = np.dot(sys_control_truncated.Ti, sys.x0)

    print "============\nEVALUATIONS\n==============="

    timeSteps = list(np.linspace(0, 1, 100))

    print "unreduced system"
    with Timer():
        Y = sys(timeSteps)

    print "system reduced with balanced truncation, auto sized"
    with Timer():
        Y_auto_truncated = sys_auto_truncated(timeSteps)

    print 'system reduced with balanced truncation, k={}'.format(k)
    with Timer():
        Y_balanced_truncated = sys_balanced_truncated(timeSteps)

    print "system reduced with controllability gramian"
    with Timer():
        Y_control_truncated = sys_control_truncated(timeSteps)

    eps_auto_truncated = [linalg.norm(y-yhat, ord=np.inf)
                          for y, yhat
                          in  zip(Y, Y_auto_truncated)]
    eps_balanced_truncated = [linalg.norm(y-yhat, ord=np.inf)
                              for y, yhat
                              in  zip(Y, Y_balanced_truncated)]
    eps_control_truncated = [linalg.norm(y-yhat, ord=np.inf)
                             for y, yhat
                             in  zip(Y, Y_control_truncated)]

    print "The original system has order ", sys.order
    print "The auto-sized system has order ", sys_auto_truncated.order
    print "and a total error of ", max(eps_auto_truncated)
    print "The balanced and truncated system has order ", \
        sys_balanced_truncated.order
    print "and a total error of ", max(eps_balanced_truncated)
    print "The control truncated system has order ", sys_control_truncated.order
    print "and a total error of ", max(eps_control_truncated)

    raise Exception
