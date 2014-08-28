from __future__ import division, print_function
import random
import math
import numpy as np
from scipy import linalg
from matplotlib.pyplot import plot, subplot, legend, figure, semilogy
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import example2sys as e2s
import pod
import time

font_options = {}


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
            Y.append(sys(t, u))
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


def systemsToReduce(k_bal_trunc, k_cont_trunc):
    red_sys = []

    for k in k_bal_trunc:
        if k:
            with_k_str = "\nwith k = {}".format(k)
        else:
            with_k_str = ""
        red_sys.append({"name": "balanced truncation" + with_k_str,
                        "shortname": "BT",
                        "reduction": "truncation_square_root_trans_matrix",
                        "k": k})

    for k in k_cont_trunc:
        with_k_str = "\nwith k = {}".format(k)
        red_sys.append({"name": "controllability truncation" + with_k_str,
                        "shortname": "CT",
                        "reduction": "controllability_truncation",
                        "k": k})

    return red_sys


def reducedAnalysis2D(unred_sys, control, k=10, k2=None,
                      T0=0., T=1., L=1., number_of_steps=100,
                      picture_destination=
                      "../plots/plot_{}_t{:.2f}_azim_{}.png"):
    print("REDUCTIONS\n--------------")

    k_bal_trunc = [None, k]
    k_cont_trunc = [k2] * (k2 is not None) + [k]

    red_sys = systemsToReduce(k_bal_trunc, k_cont_trunc)

    red_sys = reduce(unred_sys[0]["sys"], red_sys)

    print("============\nEVALUATIONS\n===============")

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
        print("and a maximal error of {}".format(max(system["eps"])))
        print("and an error at t=T of {}".format(system["eps"][-1]))

    print("==============\nPLOTS\n==============")

    figure(2)
    for system in systems[1:]:
        plot(timeSteps, system["eps"], label=system["name"])
    legend(loc="upper left")

    fig = figure()

    number_of_outputs = len(Y[0])

    X, Y = [], []
    for i in range(len(timeSteps)):
        X.append([timeSteps[i] for _ in range(number_of_outputs)])
        Y.append([j*L/(number_of_outputs-1) for j in range(number_of_outputs)])

    axes = []

    for system in range(len(systems)):
        axes.append(fig.add_subplot(221+system+10*(len(systems) > 4),
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

    save_figures = raw_input("Do you want to save the figures? (y/N) ")

    if save_figures == "y":
        for ii in xrange(360, 0, -10):
            for ax in axes:
                ax.azim = ii
            fig.savefig(picture_destination.format(control, T, axes[0].azim))

    return systems


def controllableHeatSystemComparison(N=1000, k=None, k2=None,
                                     r=0.05, T0=0., T=1., L=1.,
                                     number_of_steps=100,
                                     control="sin",
                                     integrator="dopri5",
                                     integrator_options={}):
    if k is None:
        k = max(1, int(N/50))

    print("SETUP\n====================")

    unred_sys = [{"name": "Controllable heat equation"}]

    print(unred_sys[0]["name"])
    with Timer():
        unred_sys[0]["sys"] = e2s.controllableHeatSystem(N=N, L=L,
                                                         control=control)
        unred_sys[0]["sys"].integrator = integrator
        unred_sys[0]["sys"].integrator_options = integrator_options

    pic_path = "../plots/controllable_heat_{}_t{:.2f}_azim_{}.png"
    reducedAnalysis2D(unred_sys, control, k, k2, T0, T, L, number_of_steps,
                      picture_destination=pic_path)


def optionPricingComparison(N=1000, k=None,
                            option="put", r=0.05, T=1., K=10., L=None,
                            integrator="dopri5", integrator_options={}):
    if k is None:
        k = max(1, int(N/50))
    if L is None:
        L = 3 * K

    print("SETUP\n====================")

    unred_sys = [{"name": ("Heat equation for {} option pricing" +
                           " with n = {}").format(option, N)}]

    print(unred_sys[0]["name"])
    with Timer():
        unred_sys[0]["sys"] = e2s.optionPricing(N=N, option=option,
                                                r=r, T=T, K=K, L=L)
        unred_sys[0]["sys"].integrator = integrator
        unred_sys[0]["sys"].integrator_options = integrator_options

    print("REDUCTIONS\n--------------")

    k_bal_trunc = [None, k]
    k_cont_trunc = + [k]

    red_sys = systemsToReduce(k_bal_trunc, k_cont_trunc)

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

    for system in range(6):
        axes.append(fig.add_subplot(221+system, projection='3d'))

        Z = []
        for i in range(len(timeSteps)):
            Z.append(list(systems[system]["Y"][i])[:N2])

        axes[-1].plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False)
        axes[-1].set_title(systems[system]["name"], **font_options)
        axes[-1].set_xlabel("t", **font_options)
        axes[-1].set_ylabel("K", **font_options)
        axes[-1].set_zlabel("Lambda(K)", **font_options)

    for ax in axes:
        ax.azim = 26
    fig.savefig("../plots/{}_option_azim_{}.png".format(option, axes[0].azim))

    figure(2)
    for system in systems[1:]:
        plot(timeSteps, system["eps"], label=system["name"])
    legend(loc="upper left")


def thermalRCNetworkComparison(R=1e90, C=1e87, n=100, k=10, k2=28, r=3,
                               T0=0., T=1., omega=math.pi, number_of_steps=100,
                               control="sin", input_scale=1.,
                               integrator="dopri5",
                               integrator_options={}):
    u = lambda t, x=None: np.array([e2s.simple_functions[control](omega*t)])

    print("===============\nSETUP\n===============")

    unred_sys = [{"name": "Thermal RC Netwok with n = {}".format(n)}]

    print(unred_sys[0]["name"])
    with Timer():
        C0, unred_sys[0]["sys"] = e2s.thermalRCNetwork(R, C, n, r, u,
                                                       input_scale=input_scale)
        unred_sys[0]["sys"].integrator = integrator
        unred_sys[0]["sys"].integrator_options = integrator_options

    reducedAnalysis1D(unred_sys, k, k2, T0, T, number_of_steps)


def reducedAnalysis1D(unred_sys, k=10, k2=28,
                      T0=0., T=1., number_of_steps=100):
    print("REDUCTIONS\n--------------")

    k_bal_trunc = [None, k]
    k_cont_trunc = [k2] * (k2 is not None) + [k]

    red_sys = systemsToReduce(k_bal_trunc, k_cont_trunc)

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
        print("and a maximal error of {}".format(max(system["eps"])))
        print("and an error at t=T of {}".format(system["eps"][-1]))

    print("==============\nPLOTS\n==============")

    figure(1)
    for system in systems:
        plot(timeSteps, system["Y"], label=system["name"])
    legend(loc="lower right")

    figure(2)
    for system in systems[1:4]:
        subplot(1, 2, 1)
        plot(timeSteps, system["eps"], label=system["name"])
    legend(loc="upper left")
    for system in systems[4:]:
        subplot(1, 2, 2)
        plot(timeSteps, system["eps"], label=system["name"])
    legend(loc="upper left")

    markers = ['o', 'v', '*', 'x', 'd']

    figure(3)
    for system, marker in zip(systems[1:], markers):
        sv = list(system["sys"].hsv)
        semilogy(range(len(sv)), sv,
                 marker=marker, label=system["name"])
    legend(loc="lower left")

    return systems


def loadHeat(k=10, k2=28, T0=0., T=1., number_of_steps=100,
             control="sin", omega=math.pi, control_scale=1.,
             all_state_vars=False,
             integrator="dopri5",
             integrator_options={}):
    u = lambda t, x=None: np.array([e2s.simple_functions[control](omega*t) *
                                    control_scale])

    unred_sys = [{"name": "Heat equation from\nthe SLICOT benchmarks"}]

    print(unred_sys[0]["name"])
    with Timer():
        unred_sys[0]["sys"] = e2s.example2sys("heat-cont.mat")
        unred_sys[0]["sys"].control = u
        unred_sys[0]["sys"].integrator = integrator
        unred_sys[0]["sys"].integrator_options = integrator_options

    if all_state_vars:
        unred_sys[0]["sys"].C = np.eye(unred_sys[0]["sys"].order)

    return unred_sys


def compareHeat(k=10, k2=28, T0=0., T=10., number_of_steps=300,
                control="sin", omega=math.pi, control_scale=1.,
                integrator="dopri5",
                integrator_options={}):
    unred_sys = loadHeat(k, k2, T0, T, number_of_steps,
                         control, omega, control_scale,
                         False,
                         integrator, integrator_options)

    reducedAnalysis1D(unred_sys, k, k2, T0, T, number_of_steps)


def compareHeatStates(k=10, k2=37, T0=0., T=10., number_of_steps=300,
                      control="sin", omega=math.pi, control_scale=1.,
                      integrator="dopri5",
                      integrator_options={}):
    unred_sys = loadHeat(k, k2, T0, T, number_of_steps,
                         control, omega, control_scale,
                         True,
                         integrator, integrator_options)

    L = 1.

    pic_path = "../plots/slicot_heat_{}_t{:.2f}_azim_{}.png"

    reducedAnalysis2D(unred_sys, control, k, k2, T0, T, L, number_of_steps,
                      picture_destination=pic_path)


def reduce(sys, red_sys):
    for system in red_sys:
        print(system["name"])
        with Timer():
            system["sys"] = \
                pod.lss(sys,
                        reduction=system.get("reduction", None),
                        k=system.get("k", None))

        system["error_bound"] = system["sys"].hsv[0] * \
            np.finfo(float).eps * sys.order

    return red_sys


def tableFormat(systems, solving_time=False, hankel_norm=False, min_tol=False):
    th = ("Reduction "
          " &  Order"
          " & \\specialcell[r]{Max. Error\\\\at $0\\leq t \\leq T$}"
          " & \\specialcell[r]{Max. Error\\\\at $t=T$}")

    tb_template = ("\\\\\\hline\n{:2s}"
                   " & {:3d}"
                   " & {:9.2e}"
                   " & {:9.2e}")

    if solving_time:
        th += " & \\specialcell[r]{Solving Time}"
        tb_template += " & {:9.2e}"

    if hankel_norm:
        th += " & Hankel Norm"
        tb_template += " & {:9.2e}"

    if min_tol:
        th += " & Minimal Tolerance"
        tb_template += " & {:9.2e}"

    tb = []
    for system in systems:
        results = [
            system.get("shortname", "Original"),
            system["sys"].order
            ]
        if "eps" in system:
            results.append(max(system["eps"]))
            results.append(system["eps"][-1])

        if solving_time:
            results.append(0.)

        if hankel_norm:
            results.append(system["sys"].hsv[0])

        if min_tol:
            results.append(system["error_bound"])

        tb.append(tb_template.format(*results))

    table = th
    for line in tb:
        table += line

    print(table)
