from __future__ import division, print_function
import random
import math
import numpy as np
from scipy import linalg
from matplotlib.pyplot import plot, subplot, legend, figure
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import example2sys as e2s
import pod
import time

font_options = {}

def runAnalysis(n, k, N=1, example='butter', T=20, sigma=1., integrator='dopri5'):
    sys = e2s.example2sys(example + '_' + str(n) + '.mat')
    rsys = pod.lss(sys, reduction="controllability_truncation", k=k)

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
    U = [np.array([random.gauss(0., sigma)]) for t in timeSteps]

    sys.integrator = integrator
    rsys.integrator = integrator

    sys.setupODE()
    rsys.setupODE()

    print('System of order {}'.format(sys.order))
    with Timer():
        Y = []
        for t, u in zip(timeSteps, U):
            Y.append(sys(t,u))
    print('System of order {}'.format(rsys.order))
    with Timer():
        Yhat = []
        for t, u in zip(timeSteps, U):
            Yhat.append(sys(t, u))

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

def controllableHeatSystemComparison(N=1000, k=None,
                            r=0.05, T=1., L=1.,
                            control="sin",
                            integrator="dopri5", integrator_options={}):
    if k is None:
        k = max(1,int(N/50))

    print("SETUP\n====================")

    unred_sys = [{"name" : "Controllable heat equation"}]

    print(unred_sys[0]["name"])
    with Timer():
        unred_sys[0]["sys"] = e2s.controllableHeatSystem(N=N, L=L,
                                                         control=control)
        unred_sys[0]["sys"].integrator = integrator
        unred_sys[0]["sys"].integrator_options = integrator_options

    sys = unred_sys[0]["sys"]

    print("REDUCTIONS\n--------------")

    red_sys = [{"name" : "auto truncated ab09ax",
                    "reduction" : "truncation_square_root_trans_matrix"},
               {"name" : "balanced truncated ab09ax with k = {}".format(k),
                    "reduction" : "truncation_square_root_trans_matrix",
                    "k" : k},
               {"name" : "controllability gramian reduction with k={}".format(k),
                    "reduction" : "controllability_truncation",
                    "k" : k}]

    if sys.x0 is None:
        red_sys += [{"name" : "auto truncated ab09ad",
                        "reduction" : "truncation_square_root"},
                    {"name" : "balanced truncated ab09ad with k = {}".format(k),
                        "reduction" : "truncation_square_root",
                        "k" : k}]

    red_sys = reduce(unred_sys[0]["sys"], red_sys)

    print("============\nEVALUATIONS\n===============")

    timeSteps = list(np.linspace(0, T, 100))
    systems = unred_sys + red_sys

    for system in systems:
        print(system["name"])
        with Timer():
            system["Y"] = system["sys"](timeSteps)
    
    print("===============\nERRORS\n===============")

    norm_order = np.inf

    Y = systems[0]["Y"]

    for system in systems:
        print(system["name"], "has order", system["sys"].order)
        system["eps"] = [linalg.norm(y-yhat, ord=norm_order)
                         for y, yhat in zip(Y, system["Y"])]
        print("and a maximal error of", max(system["eps"]))

    print("==============\nPLOTS\n==============")

    fig = figure()

    number_of_outputs = len(Y[0])
    
    X, Y = [], []
    for i in range(len(timeSteps)):
        X.append([timeSteps[i] for _ in range(number_of_outputs)])
        Y.append([j*L/(number_of_outputs-1) for j in range(number_of_outputs)])

    axes = []

    for system in range(len(systems)):
        axes.append(fig.add_subplot(221+system+10*(len(systems)>4),
                                  projection='3d'))

        Z = []
        for i in range(len(timeSteps)):
            Z.append(list(systems[system]["Y"][i]))

        axes[-1].plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        axes[-1].set_title(systems[system]["name"])
        axes[-1].set_xlabel("t")
        axes[-1].set_ylabel("l")
        axes[-1].set_zlabel("temperature")


    for ii in xrange(360, 0, -10):
        for ax in axes:
            ax.azim = ii
        fig.savefig("../plots/controllable_heat_{}_t{:.2f}_azim_{}.png".format(control, T, axes[0].azim))

    figure(2)
    for system in systems[1:]:
        plot(timeSteps, system["eps"], label=system["name"])
    legend(loc="upper left")

def optionPricingComparison(N=1000, k=None,
                            option="put", r=0.05, T=1., K=10., L=None,
                            integrator="dopri5", integrator_options={}):
    if k is None:
        k = max(1,int(N/50))
    if L is None:
        L = 3 * K

    print("SETUP\n====================")

    unred_sys = [{"name" : ("Heat equation for {} option pricing" +
                    " with n = {}").format(option,N)}]

    print(unred_sys[0]["name"])
    with Timer():
        unred_sys[0]["sys"] = e2s.optionPricing(N=N, option=option,
                                                r=r, T=T, K=K, L=L)
        unred_sys[0]["sys"].integrator = integrator
        unred_sys[0]["sys"].integrator_options = integrator_options

    sys = unred_sys[0]["sys"]

    print("REDUCTIONS\n--------------")

    red_sys = [{"name" : "auto truncated ab09ax",
                    "reduction" : "truncation_square_root_trans_matrix"},
               {"name" : "balanced truncated ab09ax with k = {}".format(k),
                    "reduction" : "truncation_square_root_trans_matrix",
                    "k" : k},
               {"name" : "controllability gramian reduction with k={}".format(k),
                    "reduction" : "controllability_truncation",
                    "k" : k}]

    red_sys = reduce(unred_sys[0]["sys"], red_sys)

    print("============\nEVALUATIONS\n===============")

    timeSteps = list(np.linspace(0, T, 30))
    systems = unred_sys + red_sys

    for system in systems:
        print(system["name"])
        with Timer():
            system["Y"] = system["sys"](timeSteps)
    
    print("===============\nERRORS\n===============")

    norm_order = np.inf

    Y = systems[0]["Y"]

    for system in systems:
        print(system["name"], "has order", system["sys"].order)
        system["eps"] = [0.] + [linalg.norm(y-yhat, ord=norm_order)
                                for y, yhat in zip(Y[1:], system["Y"][1:])]
        print("and a maximal error of", max(system["eps"]))

    print("==============\nPLOTS\n==============")

    fig = figure(1)
    
    N2 = int(1.5*K*N/L)
    X, Y = [], []
    for i in range(len(timeSteps)):
        X.append([timeSteps[i] for _ in range(N2)])
        Y.append([j*L/N for j in range(N2)])


    axes = []

    for system in range(4):
        axes.append(fig.add_subplot(221+system, projection='3d'))

        Z = []
        for i in range(len(timeSteps)):
            Z.append(list(systems[system]["Y"][i])[:N2])

        axes[-1].plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        axes[-1].set_title(systems[system]["name"], **font_options)
        axes[-1].set_xlabel("-t", **font_options)
        axes[-1].set_ylabel("K", **font_options)
        axes[-1].set_zlabel("Lambda(K)", **font_options)

    for ax in axes:
        ax.azim = 26
    fig.savefig("../plots/{}_option_azim_{}.png".format(option, axes[0].azim))

    figure(2)
    for system in systems[1:]:
        plot(timeSteps, system["eps"], label=system["name"])
    legend(loc="upper left")

def thermalRCNetworkComparison(R=1e90, C=1e87, n=100, k=10, r=3,
                               T0=0., T=1., omega=math.pi, number_of_steps=100,
                               integrator="dopri5",
                               integrator_options={}):
    u = lambda t, x=None: np.array([math.sin(omega*t)])

    print("===============\nSETUP\n===============")

    unred_sys = [{"name" : "Thermal RC Netwok with n = {}".format(n)}]

    print(unred_sys[0]["name"])
    with Timer():
        C0, unred_sys[0]["sys"] = e2s.thermalRCNetwork(R, C, n, r, u)
        unred_sys[0]["sys"].integrator = integrator
        unred_sys[0]["sys"].integrator_options = integrator_options

    print("REDUCTIONS\n--------------")

    red_sys = [{"name" : "auto truncated ab09ax",
                    "reduction" : "truncation_square_root_trans_matrix"},
               {"name" : "auto truncated ab09ad",
                    "reduction" : "truncation_square_root"},
               {"name" : "balanced truncated ab09ax with k = {}".format(k),
                    "reduction" : "truncation_square_root_trans_matrix",
                    "k" : k},
               {"name" : "balanced truncated ab09ad with k = {}".format(k),
                    "reduction" : "truncation_square_root",
                    "k" : k},
               {"name" : "controllability gramian reduction with k={}".format(k),
                    "reduction" : "controllability_truncation",
                    "k" : k}]

    red_sys = reduce(unred_sys[0]["sys"], red_sys)

    print("===============\nEVALUATIONS\n===============")

    timeSteps = list(np.linspace(T0, T, number_of_steps))
    systems = unred_sys + red_sys

    for system in systems:
        print(system["name"])
        with Timer():
            system["Y"] = system["sys"](timeSteps)


    print("===============\nERRORS\n===============")

    norm_order = np.inf

    Y = systems[0]["Y"]

    for system in systems:
        print(system["name"], "has order", system["sys"].order)
        system["eps"] = [linalg.norm(y-yhat, ord=norm_order)
                         for y, yhat in zip(Y, system["Y"])]
        print("and a maximal error of", max(system["eps"]))

    print("==============\nPLOTS\n==============")

    figure(1)
    for system in systems:
        plot(timeSteps, system["Y"], label=system["name"])
    legend(loc="upper left")

    figure(2)
    for system in systems[1:4]:
        subplot(1, 2, 1)
        plot(timeSteps, system["eps"], label=system["name"])
    legend(loc="upper left")
    for system in systems[4:]:
        subplot(1, 2, 2)
        plot(timeSteps, system["eps"], label=system["name"])
    legend(loc="upper left")


def reduce(sys, red_sys):
    for system in red_sys:
        print(system["name"])
        with Timer():
            system["sys"] = \
                pod.lss(sys,
                        reduction=system.get("reduction", None),
                        k=system.get("k", None))

    return red_sys
