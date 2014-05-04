from __future__ import division
import numpy as np
import scipy as sp
from pod import *
import unittest

class testTrivialCases(unittest.TestCase):
    """test trivial cases like all zeros or ones in very small systems"""
    def testIdentity(self):
        N = 5
        T = range(N)
        U = [0., 1., 0.,  2., -1.]
        R = [0., 5., 5., 15., 10.]
        R = [np.array(r).reshape(-1,) for r in R]

        sys = StateSpaceSystem(np.zeros((5,5)), np.ones((5,1)),
                               np.ones((1,5)), np.zeros((1,1)))
        sys.setupODE()

        for i in range(1, N):
            self.assertAlmostEqual(sys(T[i], U[i]), R[i])

        R = [r+u for r, u in zip(R, U)]
        R = [np.array(r).reshape(-1,) for r in R]
        sys = StateSpaceSystem(np.zeros((5,5)), np.ones((5,1)),
                               np.ones((1,5)), np.ones((1,1)))
        sys.setupODE()

        for i in range(1, N):
            self.assertAlmostEqual(sys(T[i], U[i]), R[i])

    def testStability(self):
        instableSys = StateSpaceSystem(np.zeros((5,5)), np.ones((5,1)),
                               np.ones((1,5)), np.zeros((1,1)))
        self.assertFalse(instableSys.isStable)

        stableSys = StateSpaceSystem(np.diag(np.ones(5)*(-1)), np.ones((5,1)),
                               np.ones((1,5)), np.zeros((1,1)))
        self.assertTrue(stableSys.isStable)

class testLss(unittest.TestCase):
    """test lss functionalities"""
    def test_abcd_normalize(self):
        A, B = np.zeros((5,5)), np.ones((5,1))
        C, D = np.ones((1,5)), np.zeros((1,1))
        sys_without_D = lss(A, B, C, None)
        sys_without_A = lss(None, B, C, D)

        self.assertAlmostEqual(sys_without_A.A, A)
        self.assertAlmostEqual(sys_without_D.D, D)
        


if __name__ == '__main__':
    unittest.main()