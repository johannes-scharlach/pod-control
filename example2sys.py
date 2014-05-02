import scipy.io
import pod
import random
import scipy as sp

def example2sys(filename):
    data = scipy.io.loadmat('example_data/' + filename)
    return pod.StateSpaceSystem(data['A'], data['B'], data['C'], data['D'])

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

    return pod.StateSpaceSystem(A, B, C, D)

def stableRandomSystem(*args, **kwargs):
    for i in range(1000):
        sys = generateRandomExample(*args, **kwargs)
        if sys.isStable:
            return sys
