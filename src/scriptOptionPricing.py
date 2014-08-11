from __future__ import division, print_function
import random
import math
import numpy as np
from scipy import linalg
from matplotlib.pyplot import plot, subplot, legend, figure, show, suptitle, \
        xlabel, ylabel
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import example2sys as e2s
import pod
import time
from analysis import *

N = 1000
k = 20
k2 = 62
integrator = "dopri5"
integrator_options = {}
T = 1.
L = 30
K = 10.5
r = 0.05
option = "put"

font_options = {}


print("SETUP\n====================")

unred_sys = [{"name" : ("Heat equation for {} option pricing" +
                " with n = {}").format(option, N)}]

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
           {"name" : "controllability gramian reduction with k={}".format(k2),
                "reduction" : "controllability_truncation",
                "k" : k2},
           {"name" : "controllability gramian reduction with k={}".format(k),
                "reduction" : "controllability_truncation",
                "k" : k}]

red_sys = reduce(unred_sys[0]["sys"], red_sys)

print("============\nEVALUATIONS\n===============")

timeSteps = list(np.linspace(0., T, 30))
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
                     for y, yhat in zip(Y[:], system["Y"][:])]
    print("and a maximal error of", max(system["eps"]))

print("==============\nPLOTS\n==============")

fig = figure()

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
    axes[-1].set_ylabel("S", **font_options)
    axes[-1].set_zlabel("f", **font_options)

for ax in axes:
    ax.azim = 26
# fig.savefig("../plots/{}_option_azim_{}.png".format(option, axes[0].azim),
#             bbox_inches="tight")

fig = figure()
for system in systems[1:]:
    plot(timeSteps[:], system["eps"], label=system["name"])
legend(loc="upper right")
xlabel("t")
ylabel("Error")

# fig.savefig("../plots/{}_option_pricing_errors.png".format(option),
#             bbox_inches="tight")

fig = figure()
for system in systems[1:]:
    plot([j*L/N for j in range(N2)],
         systems[0]["Y"][0][:N2]-system["Y"][0][:N2],
         label=system["name"])
legend(loc="upper right")
xlabel("S")
ylabel("Error")

# fig.savefig("../plots/{}_option_pricing_errors_t0.png".format(option),
#             bbox_inches="tight")

show()
