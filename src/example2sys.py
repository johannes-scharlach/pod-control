from __future__ import division
import math

import scipy.io
import scipy.sparse
import pod
import random
import numpy as np


def BLaC(x, delta=0.5):
    xMod = math.fmod(x, 2)
    return xMod/delta*(xMod < delta) + 1*(xMod >= delta)*(xMod <= 2-delta) \
        + (2-xMod)/delta*(xMod > 2-delta)

simple_functions = {
    "sin": lambda x: math.sin(x),
    "cos": lambda x: math.cos(x),
    "zero": lambda x: 0.,
    "one": lambda x: 1.,
    "identity": lambda x: x,
    "hat": lambda x: (math.fmod(x+1, 2.)-1) *
    (1-2*(math.fmod(x+1, 4.) > 2)),
    "BLaC": BLaC,
    "BLaCsteep": lambda x: BLaC(x, delta=0.1)}


def _todense_float(A):
    if scipy.sparse.issparse(A):
        return A.toarray()*1.
    elif A is None:
        return None
    return np.asanyarray(A, dtype=np.float)


def example2sys(filename):
    data = scipy.io.loadmat('example_data/' + filename)
    A = data['A']
    B = data['B']
    C = data.get('C', B.transpose())
    D = data.get('D', None)
    return pod.lss(*map(_todense_float, [A, B, C, D]))


def heatSystemA(sizes, h2, alpha):
    """Build matrix A for a heat equation"""
    if len(sizes) != 1:
        e = "dimensions other than 1 not yet implemented"
        raise NotImplementedError(e)
    N = sizes[0]
    h2 = h2[0]
    alpha = alpha[0]
    main_diagonal = np.ones(N-1) * (-2 / h2) * alpha
    secondary_diagonal = np.ones(N-2) / h2 * alpha

    return np.diag(main_diagonal) + np.diag(secondary_diagonal, 1) + \
        np.diag(secondary_diagonal, -1)


def controllableHeatSystem(N, alpha=1., L=1., control="sin"):
    if N % 2:
        raise ValueError("N-1 has to be odd, to control the middle")

    scale = 0.0007
    input_scale = scale * 120

    h2 = (L/N)**2

    A = heatSystemA([N], [h2], [alpha])

    B = np.zeros((N-1, 1))
    B[N/2-1][0] = input_scale

    C = np.concatenate((np.zeros((1, N-1)), np.eye(N-1), np.zeros((1, N-1))))
    # C = np.zeros((5, N-1))
    # C[1][1*N/4-1] = 1.
    # C[2][2*N/4-1] = 1.
    # C[3][3*N/4-1] = 1.

    D = None

    x0 = np.array([min(i+1, N-1-i) * scale / (N/2) for i in range(N-1)])
    x0 = None

    sys = pod.lss(A, B, C, D)

    sys.x0 = x0
    sys.control = lambda t, x=None: np.array([simple_functions[control](t)])

    return sys


def heatSystem(N, g0_scale=None, gN_scale=None, alpha=1., L=1.0):
    """Generate a state space system that solves the heat equation.

    This sets up the matrices `A`, `B`, `C` and `D` where `B` is scaled in a
    way that the input (boundary conditions) should have an average norm of 1.
    This allows for proper reduction of the model. Boundary conditions and
    start vector need to be set separately.

    """
    inputs = (g0_scale is not None) + (gN_scale is not None)

    h2 = (L/N)**2

    A = heatSystemA([N], [h2], [alpha])

    B = np.zeros((N-1, inputs))
    if g0_scale:
        B[0][0] = 1 / h2 * g0_scale
    if gN_scale:
        B[-1][-1] = 1 / h2 * gN_scale
    C = np.concatenate((np.zeros((1, N-1)), np.eye(N-1), np.zeros((1, N-1))))
    D = np.zeros((N+1, inputs))
    if g0_scale:
        D[0][0] = g0_scale
    if gN_scale:
        D[-1][-1] = gN_scale

    sys = pod.lss(A, B, C, D)

    return sys


def optionPricing(N=1000, option="put", r=0.05, T=1., K=100., L=None):
    """Generate a State-Space System for the heat eqation for option pricing

    """

    scaled = True

    if not L:
        L = 10 * K
    h = L/N

    g0_scale = 1
    gN_scale = 1

    if option is "put":
        def boundary_conditions(t, y=None):
            a = math.e**(-r*(t)) * (1 * scaled + (not scaled) * K)
            return np.array([a]) if scaled else np.array([a, 0.])
        if scaled:
            g0_scale = K
            gN_scale = None
        x0 = [max(K-h*i, 0) for i in range(1, N)]
    elif option is "call":
        def boundary_conditions(t, y=None):
            b = 1 - math.e**(-r*(t))*K * (1 * (not scaled) + scaled / L)
            return np.array([b]) if scaled else np.array([0., b])
        if scaled:
            g0_scale = None
            gN_scale = L
        x0 = [max(h*i-K, 0) for i in range(1, N)]
    else:
        raise ValueError("No such option aviable")

    sys = heatSystem(N, L=L, g0_scale=g0_scale, gN_scale=gN_scale)
    sys.control = boundary_conditions
    sys.x0 = np.array(x0)

    return sys


def rcLadder(resistors, capacitors, input_scale=1., outputs=[-1]):
    """A system of a rcLadder

    Parameters
    ----------
        resistors : iterable
            The resistors ``R_1`` until ``R_N`` or
            ``R_[N+1]`` that are between the points ``e_[i-1]``
            and ``e_i.``
        capacitors : iterable
            The capacities ``C_i`` between the points ``e_i`` and
            the earth.
        input_scale : decimal, optional
            The scaling that is needed to keep the norm of the input function
            of the system equal to one.
        outputs : list, optional
            List of the outputs indices that should be returned. Default is
            ``[-1]`` so that the last value gets returned.

    Returns
    -------
        sys : pod.lss
            The linear state space system that solves the problem with input
            function ``e_0``
    """
    N = len(capacitors)

    conductivities = [1./R for R in resistors]
    if len(conductivities) == N:
        conductivities.append(0.)
    conductivities = np.array(conductivities)

    capacitors = np.array(capacitors)

    main_diagonal = - (conductivities[:-1] + conductivities[1:]) / capacitors
    left_diagonal = conductivities[1:-1] / capacitors[1:]
    right_diagonal = conductivities[1:-1] / capacitors[:-1]

    A = np.diag(main_diagonal) + np.diag(right_diagonal, 1) + \
        np.diag(left_diagonal, -1)
    B = np.zeros((N, 1))
    B[0][0] = conductivities[0]/capacitors[0] * input_scale
    C = np.zeros((len(outputs), N))
    for i in range(len(outputs)):
        C[i][outputs[i]] = 1.
    D = None

    return pod.lss(A, B, C, D)


def thermalRCNetwork(R, C, n, r, u, input_scale=1.):
    capacitors = _thermalRCNetworkCapacitors(C, n, r)
    resistors = _thermalRCNetworkResistors(C, 0., R, n, r)

    sys = rcLadder(resistors, capacitors[1:], outputs=[0],
                   input_scale=input_scale)

    sys.control = u

    return capacitors[0], sys


def _thermalRCNetworkCapacitors(C, n, r):
    return [(r-1)*(r**i)/(r**n-1)*C for i in range(n)]


def _thermalRCNetworkResistors(C, C0, R, n, r):
    Rs = C / (C+C0) * R
    resistors = [(2+r)*(r-1)*.5] + [r**i + r**(i+1) for i in range(1, n-1)]
    resistors = np.array(resistors) * (1/(r**n-1)*Rs)
    resistors = list(resistors) + [(3**(n-1) / (3**n - 1) - 1) * Rs + R]
    return list(resistors)


def _neg(x):
    if x < 0.:
        return -x
    return x


def generateRandomExample(n, m, p=None,
                          distribution=random.gauss,
                          distributionArguments=[0., 1.]):
    """Generate a random example of arbitraty order

    Parameters
    ----------
        n : int
            system order
        m : int
            number of Inputs
        p : int
            number of outputs ``[p=m]``
        distribution : callable
            distribution of the matrix values [distribution=gauss(0., 1.)]

    Returns
    -------
        sys : pod.lss
            random StateSpaceSystem with the parameters set in the input

    """
    if p is None:
        p = m

    A = [[_neg(distribution(*distributionArguments)*(i >= j))
          for i in range(n)]
         for j in range(n)]
    B = [[distribution(*distributionArguments) for i in range(m)]
         for j in range(n)]
    C = [[distribution(*distributionArguments) for i in range(n)]
         for j in range(p)]
    D = [[distribution(*distributionArguments) for i in range(m)]
         for j in range(p)]

    return pod.lss(A, B, C, D)


def stableRandomSystem(*args, **kwargs):
    for i in range(1000):
        sys = generateRandomExample(*args, **kwargs)
        if pod.isStable(sys.A):
            return sys
