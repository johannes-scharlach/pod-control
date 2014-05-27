from __future__ import division
import math

import scipy.io
import pod
import random
import scipy as sp
import numpy as np

def example2sys(filename):
    data = scipy.io.loadmat('example_data/' + filename)
    return pod.lss(data['A'], data['B'], data['C'], data['D'])

def heatSystem(N, L=1.0, g0_scale=None, gN_scale=None):

    inputs = (g0_scale is not None) + (gN_scale is not None)

    h2 = (L/N)**2
    main_diagonal = np.ones(N-1) * (-2 * h2)
    secondary_diagonal = np.ones(N-2) * h2

    A = np.diag(main_diagonal) + np.diag(secondary_diagonal, 1) + \
            np.diag(secondary_diagonal, -1)
    B = np.zeros((N-1,inputs))
    if g0_scale:
        B[0][0] = h2 * g0_scale
    if gN_scale:
        B[-1][-1] = h2 * gN_scale
    C = np.concatenate((np.zeros((1,N-1)), np.eye(N-1), np.zeros((1,N-1))))
    D = np.zeros((N+1,inputs))
    if g0_scale:
        D[0][0] = 1
    if gN_scale:
        D[-1][-1] = 1

    sys = pod.lss(A, B, C, D)

    return sys

def optionPricing(N=1000, option="put", r=0.05, T=1., K=100., L=None):
    """Generates a State-Space System for the heat eqation for option pricing

    """
    if not L:
        L = 10 * K
    h = L/N

    if option is "put":
        def boundary_conditions(t, y=None):
            a = math.e**(-r*(-t))
            return np.array([a])
        g0_scale = K
        gN_scale = None
        x0 = [max(K-h*i, 0) for i in range(1,N)]
    elif option is "call":
        def boundary_conditions(t, y=None):
            b = 1 - math.e**(-r*(-t))*K/L
            return np.array([b])
        g0_scale = None
        gN_scale = L
        x0 = [max(h*i-K, 0) for i in range(1,N)]
    else:
        raise ValueError("No such option aviable")

    sys = heatSystem(N, L=L, g0_scale=g0_scale, gN_scale=gN_scale)
    sys.control = boundary_conditions
    sys.x0 = np.array(x0)

    return sys

def generateRandomExample(n, m, p=None,
        distribution=random.gauss, distributionArguments=[0., 1.]):
    """Generate a random example of arbitraty order

    How to use:
        sys = generateRandomExample(n, m, [p, distribution])

    Inputs:
        n   system order
        m   number of Inputs
        p   number of outputs [p=m]
        distribution
            distribution of the matrix values [distribution=gauss(0., 1.)]

    Output:
        sys random StateSpaceSystem with the parameters set in the input
    """

    if p is None:
        p = m

    A = [[distribution(*distributionArguments) for i in range(n)]
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
