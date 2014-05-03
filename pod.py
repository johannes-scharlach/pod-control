from __future__ import division
import math
import numpy as np
import scipy as sp
from numpy import array
from scipy.integrate import ode
# from scipy.signal.ltisys import abcd_normalize
# currently buggy. Will be fixed in scipy v0.15
# instead the following can be used temporarily
from futurescipy import abcd_normalize


class StateSpaceSystem(object):
    """A linear state Space system that can be called in order to solve for
    a particular input and time
    """
    def __init__(self, A, B, C, D=[[]]):
        self.A = array(A)
        self.B = array(B)
        self.C = array(C)
        self.D = array(D)

    def setupODE(self, x0=0, t0=0.0, integrator='dopri5', **options):
        self.state = ode(self.f, jac=self.jac)
        self.state.set_integrator(integrator,**options)
        if x0 is 0:
            x0 = array([0. for _ in range(self.order)])
        self.state.set_initial_value(x0, t0)

    #@InputOfCalls
    def f(self, t, y, u):
        if callable(u):
            u = u(t=t, y=y)
        return np.dot(self.A, array(y)) + np.dot(self.B, array(u).reshape(-1,))

    def jac(self, t, y):
        return self.A

    def __call__(self, t, u):
        if isinstance(t, list):
            results = []
            for time, control in zip(t, u):
                state = self.solve(time, control)
                results.append(np.dot(self.C, state) + 
                               np.dot(self.D, array(control).reshape(-1,)))
            return results
        else:
            state = self.solve(t, u)
            return np.dot(self.C, state) + np.dot(self.D, array(u).reshape(-1,))

    @property
    def t(self):
        return self.state.t

    @property
    def x(self):
        return self.state.y

    @property
    def order(self):
        return self.A.shape[0]

    def solve(self, time, control):
        self.state.set_f_params(control)
        self.state.integrate(time)
        return self.state.y

    @property
    def isStable(self):
        D,V = np.linalg.eig(self.A)
        # print D.shape
        # print D
        for e in D:
            if e.real >= 0:
                return False
        return True

    def truncate(self, k=0, tol=0.0, balance=True, scale=False):
        """Use the square root algorithm and optionally balance the system to
        truncate it
        """
        if not self.isStable:
            raise ValueError("This doesn't seem to be a stable system!")

        try:
            from slycot import ab09ad
        except ImportError:
            raise ControlSlycot("can't find slycot subroutine ab09ad")

        if balance:
            job = 'B'
        else:
            job = 'N'

        if scale:
            equil = 'S'
        else:
            equil = 'N'

        dico = 'C'
        n = np.size(self.A,0)
        m = np.size(self.B,1)
        p = np.size(self.C,0)
        nr = k
        Nr, Ar, Br, Cr, hsv = ab09ad(dico,job,equil,n,m,p,self.A,self.B,self.C,nr,tol)
        self.hsv = hsv
   
        return StateSpaceSystem(Ar, Br, Cr, self.D)

class lss(object):
    """linear state space system.

    Will need a proper numpy-style docstring

    """
    def __init__(self, *args, **kwargs):
        """Construct a linear state space system

        Default contstructor is called like lss(A,B,C,D) and a system can
        easily be copied by calling lss(sys) where sys is a
        lss object itself.
        """

        if len(args) == 4:
            (A, B, C, D) = abcd_normalize(*args)
        elif len(args) == 1:
            A = args[0].A
            B = args[0].B
            C = args[0].C
            D = args[0].D
        else:
            raise ValueError("Needs 1 or 4 arguments; received %i." % len(args))

        if kwargs:
            raise NotImplementedError












