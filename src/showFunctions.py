from __future__ import division, print_function
import math
import numpy as np
import example2sys as e2s
from matplotlib.pylab import plot, figure, legend, ylim, xlim

T0 = 0.
T = 10.
number_of_steps = 10000
omega = math.pi

timeSteps = np.linspace(T0, T, number_of_steps)
input_vals = list(timeSteps*omega)
timeSteps = list(timeSteps)

figure()

f2p = ["zero", "one", "sin", "cos", "BLaC", "BLaCsteep"]

for key in f2p:
    Y = map(e2s.simple_functions[key], input_vals)
    plot(timeSteps, Y, label=key)

legend(loc="lower right")
ylim([-1., 1.1])
xlim([T0, T])
