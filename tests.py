"""Unit tests and functional tests for the pod.py package

"""

from __future__ import division
import numpy as np
import scipy as sp
from pod import *
import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_allclose


def _number_to_array(s):
    return np.array([s])

class testLss(unittest.TestCase):
    """test lss functionalities"""
    def test_abcd_normalize(self):
        A, B = np.zeros((5,5)), np.ones((5,1))
        C, D = np.ones((1,5)), np.zeros((1,1))
        sys_without_D = lss(A, B, C, None)
        sys_without_A = lss(None, B, C, D)

        assert_array_equal(sys_without_A.A, A)
        assert_array_equal(sys_without_D.D, D)

    def test_zero_control(self):
        A, B = np.zeros((5,5)), np.ones((5,1))
        C, D = np.ones((1,5)), np.zeros((1,1))
        sys = lss(A, B, C, D)
        sys.x0 = np.ones((5,1))
        sys(2.0)

        assert_array_equal(sys.x, np.ones((5,1)))

    def testIdentity(self):
        N = 5
        T = range(N)
        U = [0., 1., 0.,  2., -1.]
        U = map(_number_to_array, U)
        R = [0., 5., 5., 15., 10.]
        R = map(_number_to_array, R)

        A, B = np.zeros((5,5)), np.ones((5,1))
        C, D = np.ones((1,5)), None

        sys = lss(A,B,C,D)

        for i in range(1, N):
            self.assertAlmostEqual(sys(T[i], U[i]), R[i])

        R = [r+u for r, u in zip(R, U)]
        R = map(np.array, R)
        D = np.ones((1,1))

        sys = lss(A, B, C, D)

        for i in range(1, N):
            self.assertAlmostEqual(sys(T[i], U[i]), R[i])

    def test_f(self):
        A = [[ 1., 1.],
             [ 0., 1.]]
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


if __name__ == '__main__':
    unittest.main()