"""Unit tests and functional tests for the pod.py package

"""

from __future__ import division, print_function
import numpy as np
import scipy as sp
import math
from pod import *
from example2sys import *
import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_allclose


def _number_to_array(s):
    return np.array([s])

class testPod(unittest.TestCase):
    """test lss functionalities"""
    def test_abcd_normalize(self):
        A, B = np.zeros((5, 5)), np.ones((5, 1))
        C, D = np.ones((1, 5)), np.zeros((1, 1))
        sys_without_D = lss(A, B, C, None)
        sys_without_A = lss(None, B, C, D)

        assert_array_equal(sys_without_A.A, A)
        assert_array_equal(sys_without_D.D, D)

    def test_zero_control(self):
        A, B = np.zeros((5, 5)), np.ones((5, 1))
        C, D = np.ones((1, 5)), np.zeros((1, 1))
        sys = lss(A, B, C, D)
        sys.x0 = np.ones((5, 1))
        sys(2.0)

        assert_array_equal(sys.x, np.ones((5, 1)))

    def testIdentity(self):
        N = 5
        T = range(N)
        U = [0., 1., 0.,  2., -1.]
        U = map(_number_to_array, U)
        R = [0., 5., 5., 15., 10.]
        R = map(_number_to_array, R)

        A, B = None, np.ones((5, 1))
        C, D = np.ones((1, 5)), None

        sys = lss(A,B,C,D)

        for i in range(1, N):
            self.assertAlmostEqual(sys(T[i], U[i]), R[i])
            assert sys.t == T[i]

        sys.setupODE()
        timeWithSteps = [list(np.linspace(t-1, t, 2**t)) for t in T]
        for i in range(1, N):
            results = sys(timeWithSteps[i], U[i])
            self.assertAlmostEqual(results[-1], R[i])
            assert sys.t == timeWithSteps[i][-1]

        R = [r+u for r, u in zip(R, U)]
        R = map(np.array, R)
        D = np.ones((1, 1))

        sys = lss(A, B, C, D)

        for i in range(1, N):
            self.assertAlmostEqual(sys(T[i], U[i]), R[i])

    def test_f(self):
        A = [[1., 1.],
             [0., 1.]]
        B = [[1.],
             [1.]]
        C = [[1., 1.]]
        D = [[0.]]

        x = [0., 0.]
        u = [1.]

        sys = lss(A, B, C, D)

        assert_array_equal(sys.f(0., np.array(x), u), np.array([1., 1.]))

        x = [1., 1.]
        u = [0.]

        assert_array_equal(sys.f(0., np.array(x), u), np.array([2., 1.]))

        x = [10., 2.]
        u = [-3]

        assert_array_equal(sys.f(0., np.array(x), u), np.array([10+2-3., 2-3.]))

    def test_truncation_functions(self):
        """Reduce system of order 3 and check truncation matrices.

        Matrix `A` is in real schur form with all eigenvalues in the left half
        of the complex plane. The system is reduced from order 3 to orders 1 
        and 2. Order, number of inputs and outputs and the pseudo inverse
        property of T and Ti of the systems are checked.

        """
        A = np.array([[-6, -3, 1],
                      [0, -2.2, 6],
                      [0, 0, -0.5]])
        B = np.array([[1.],
                      [1.],
                      [1.]])
        C = np.array([[2., 1., 0.002]])
        D = None

        sys = lss(A, B, C, D)
        sys.x0 = np.ones((3,))

        for reduction in lss.reduction_functions:
            for k in [1, 2, 3]:
                rsys = lss(sys, reduction=reduction, k=k)

                assert rsys.order == k
                assert rsys.inputs == 1
                assert rsys.outputs == 1

                if hasattr(rsys, 'T'):
                    assert rsys.T.shape == (3, k)
                    assert rsys.Ti.shape == (k, 3)
    
                    assert_array_almost_equal(np.dot(rsys.Ti, rsys.T),
                                              np.eye(k))
                    assert_array_almost_equal(np.dot(rsys.Ti, sys.x0),
                                              rsys.x0)

class testExample2sys(unittest.TestCase):
    """Test the example system generator"""
    def test_rcLadder(self):
        resistors = [1.0, 1.0, 1.0]
        capacitors = [1.0, 1.0, 1.0]

        sys = rcLadder(resistors, capacitors)
        sys2 = rcLadder(resistors + [np.inf], capacitors)

        assert sys.inputs == 1
        assert sys.outputs == 1
        assert sys.order == 3
        for matrix in ['A', 'B', 'C', 'D']:
            assert_array_equal(getattr(sys, matrix), getattr(sys2, matrix))

    def test_thermalRCNetwork(self):
        u = lambda t, x=None: np.array([1.])
        C0, sys = thermalRCNetwork(1e90, 1e87, 100, 3, u)

        assert sys.inputs == 1
        assert sys.outputs == 1
        assert sys.order == 99

        self.assertAlmostEqual(sys.control(0.), np.array([1.0]))
        self.assertAlmostEqual(sys.control(math.pi), np.array([1.0]))
        self.assertAlmostEqual(sys.control(.5*math.pi), np.array([1.0]))

        u = lambda t, x=None: np.array([math.sin(t)])
        C0, sys = thermalRCNetwork(1e90, 1e87, 100, 3, u)

        assert sys.inputs == 1
        assert sys.outputs == 1
        assert sys.order == 99

        self.assertAlmostEqual(sys.control(0.), np.array([.0]))
        self.assertAlmostEqual(sys.control(math.pi), np.array([.0]))
        self.assertAlmostEqual(sys.control(.5*math.pi), np.array([1.0]))

if __name__ == '__main__':
    unittest.main()
    
