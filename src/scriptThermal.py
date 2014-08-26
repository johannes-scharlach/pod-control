from __future__ import division, print_function
import example2sys as e2s
import numpy as np
from matplotlib.pyplot import plot, figure
from matplotlib import cm

control = "sin"

T0 = 0.
T = 80
number_of_steps = 300
omega = 1.

R = 10.e0
deltaR = -0.01e0
C = .01e0
deltaC = 0.01e0
n = 10
r = 3
u_factor = 3.e0
u = lambda t, x=None: np.array([e2s.simple_functions[control](omega*t)]) \
    * u_factor
u_scale = 3.e10

rcValues = [{"resistors": [5.52, 17.2, 55.2, 46.8, 56.7, 27.9],
             "capacitors": [.000455, .00388, .0115, .0481, .0316, 2.79],
             "input_scale": 1.,
             "outputs": [-1]},
            {"resistors": [5.52, 17.2, 55.2, 46.8, 56.7, 27.9],
             "capacitors": [.000455, .00388, .0115, .0481, .0316, 2.79],
             "input_scale": 1.,
             "outputs": range(6)},
            {"resistors": e2s._thermalRCNetworkResistors(.000455, .00388,
                                                         5.52, 6, 3) +
             [17.2, 55.2, 46.8, 56.7, 27.9],
             "capacitors": e2s._thermalRCNetworkCapacitors(.000455, 6, 3) +
                [.00388, .0115, .0481, .0316, 2.79],
             "input_scale": 1.,
             "outputs": [-1]},
            {"resistors": e2s._thermalRCNetworkResistors(.000455, .00388,
                                                         5.52, 10, 3) +
                [17.2, 55.2, 46.8, 56.7, 27.9],
             "capacitors": e2s._thermalRCNetworkCapacitors(.000455, 10, 3) +
                [.00388, .0115, .0481, .0316, 2.79],
             "input_scale": 1.,
             "outputs": [-1]},
            {"resistors": e2s._thermalRCNetworkResistors(.000455, .00388,
                                                         5.52, 50, 3) +
                [17.2, 55.2, 46.8, 56.7, 27.9],
             "capacitors": e2s._thermalRCNetworkCapacitors(.000455, 50, 3)
                + [.00388, .0115, .0481, .0316, 2.79],
             "input_scale": 1.,
             "outputs": [-1]},
            {"resistors": [R + i * deltaR for i in xrange(n)],
             "capacitors": [C + i * deltaC for i in xrange(n)],
             "input_scale": u_scale,
             "outputs": range(n)},
            {"resistors": [R + i * deltaR for i in xrange(n)],
             "capacitors": [C + i * deltaC for i in xrange(n)],
             "input_scale": u_scale,
             "outputs": range(int(n/2), n)},
            {"resistors": [R + i * deltaR for i in xrange(n)],
             "capacitors": [C + i * deltaC for i in xrange(n)],
             "input_scale": u_scale,
             "outputs": range(n-10, n)},
            {"resistors": [R + i * deltaR for i in xrange(n)],
             "capacitors": [C + i * deltaC for i in xrange(n)],
             "input_scale": u_scale,
             "outputs": range(0, 10)}]

valueIndex = 3

resistors = rcValues[valueIndex]["resistors"]
capacitors = rcValues[valueIndex]["capacitors"]
input_scale = rcValues[valueIndex]["input_scale"]
outputs = rcValues[valueIndex]["outputs"]

sys = e2s.rcLadder(resistors, capacitors, input_scale, outputs)
sys.control = u

# C0, sys = e2s.thermalRCNetwork(R, C, n, r, u)


timeSteps = list(np.linspace(T0, T, number_of_steps))

# Y = sys(timeSteps)

# fig = figure()
# if len(outputs) is 1:
#     plot(timeSteps, Y)
# else:
#     ax = fig.add_subplot(111, projection='3d')
#     Xgrid, Ygrid = np.meshgrid(outputs, timeSteps)
#     Y = [list(y) for y in Y]
#     ax.plot_surface(Xgrid, Ygrid, Y, rstride=1, cstride=1, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)
