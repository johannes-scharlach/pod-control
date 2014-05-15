"""Model order reduction for linear state space systems can be done with
Proper Orthogonal Decomposition (POD) methods. Some of them can be simply
applied when creating a system.

"""

from __future__ import division

import collections
import numpy as np
import scipy as sp
from numpy import array
from scipy.integrate import ode
# from scipy.signal.ltisys import abcd_normalize
# currently buggy. Will be fixed in scipy v0.15
# instead the following can be used temporarily
from futurescipy import abcd_normalize


def truncation_square_root(A, B, C,
                           k=0, tol=0.0,
                           balance=True, scale=True,
                           check_stability=True,
                           length_cache_array=None):
    """Perform truncation of a system. Scaling and balancing are optional

    This allows to reduce a linear state space system by either specifying it
    to a certain number of states `k` or by specifying a error tolerance `tol`
    relative to the input. In theory the most accurate results are achieved by
    using `balance` and `scale` but the size of the error strongly depends on
    the particular problem and scaling and balancing may in some cases cost
    too much.

    Parameters
    ----------
    A : array_like
    B : array_like
    C : array_like
        State-Space matrices of the system that should be reduced
    k : int, optional
        Order of the output system
    tol : float, optional
        Error of the output system, based on the Hankel Singular Values
    balance : Boolean, optional
        Balance the system before reducing it to make sure, that the error
        is kept small
    scale : Boolean, optional
        Scale the system
    check_stability : Boolean, optional
        Checks if all the real parts of the eigenvalues of A are in the left
        half of the complex plane.

    Returns
    -------
    Nr : int
        Actual size of the system, based on the error: If the Machine error
        would have been bigger than the error of the reduced system, it may
        happen that ``Nr < k`` and if the error would be inconsiderably bad,
        It might be the case that ``Nr > k``. In case `k` was never specified,
        this is purely based on `tol`
    Ar, Br, Cr : ndarray
        Reduced arrays
    hsv : ndarray
        Hankel singular values of the original system. The size of the error
        may be calculated based on this.

    Raises
    ------
    ValueError
        If the system that's provided is not stable (i.e. `A` has eigenvalues
        which have non-negative real parts)
    ImportError
        If the slycot subroutine `ab09ad` can't be found. Occurs if the
        slycot package is not installed.

    """
    if check_stability and not isStable(A):
        raise ValueError("This doesn't seem to be a stable system!")
    try:
        from slycot import ab09ad
    except ImportError:
        raise ImportError("can't find slycot subroutine ab09ad")

    if balance:
        job = 'B'
    else:
        job = 'N'

    if scale:
        equil = 'S'
    else:
        equil = 'N'

    dico = 'C'
    n = np.size(A,0)
    m = np.size(B,1)
    p = np.size(C,0)
    nr = k

    if not length_cache_array:
        length_cache_array = 5*n**2 + n*m + n*p + 10*n + m*p

    return ab09ad(dico,job,equil,n,m,p,A,B,C,nr,tol,ldwork=length_cache_array)

def controllabilityTruncation(A,B,C,k=None,check_stability=True):
    """Truncate the system based on the controllability Gramian

    Solves the Lyapunov Equation for ``AP + PA^H + B B^H`` and computes the
    eigenvalues and eigenvectors of `P` which are used to truncate the system.
    """
    if check_stability and not isStable(A):
        raise ValueError("This doesn't seem to be a stable system!")

    if not k:
        k = int(a.shape[0]/2)

    P = sp.linalg.solve_lyapunov(A, -np.dot(B, B.H))

    Lambdak, Uk = sp.linalg.eigh(P, eigvals=(0,k), check_finite=False,
                                 overwrite_a=True, overwrite_b=True)

    UkH = Uk.H

    A = np.dot(UkH, np.dot(A, Uk))
    B = np.dot(UkH, B)
    C = np.dot(C, Uk)

    return k, A, B, C, Lambdak

def isStable(A):
    """Check if all eigenvalues are in the left half of the complex plane"""
    D, V = np.linalg.eig(A)
    return (D.real < 0).all()

class lss(object):
    """linear time independent state space system.

    Default contstructor is called like ``lss(A,B,C,D)`` and a system can
    easily be copied by calling ``lss(sys)`` where `sys` is a
    lss object itself.

    Parameters
    ----------
    A : array_like
    B : array_like
    C : array_like
    D : array_like
        State-Space matrices. If one of the matrices is None, it is
        replaced by a zero matrix with appropriate dimensions.
    reduction : {'truncation_square_root'}, optional
        Choose method of reduction. If it isn't provided, matrices are 
        used without reduction.
    \*\*reduction_options : dict, optional
        The arguments with which the reduction method is called.

    Attributes 
    ----------
    x0 : array_like, optional
        Initial state. Defaults to the zero state.
    t0 : float
        Initial value for `t`
    integrator : str
        Name of the integrator used by ``scipy.integrate.ode``
    integrator_options : dict
        Options for the specified integrator that can be set.
    reduction_functions : dict
        The functions that can be choosen as an input parameter with the
        `reduction` keyword.

    """

    x0 = None
    t0 = 0.0
    integrator = 'dopri5'
    integrator_options = {}
    reduction_functions = {'truncation_square_root' : truncation_square_root}

    def __init__(self, *create_from, **reduction_options):
        """Initialize a linear state space system

        """

        if len(create_from) == 4:
            (self.A, self.B, self.C, self.D) = abcd_normalize(*create_from)
            self.inputs = self.B.shape[1]
            self.outputs = self.C.shape[0]
            self.control = np.zeros((self.inputs,))
        elif len(create_from) == 1:
            self.A = create_from[0].A
            self.B = create_from[0].B
            self.C = create_from[0].C
            self.D = create_from[0].D
            self.inputs = create_from[0].inputs
            self.outputs = create_from[0].outputs
            self.control = create_from[0].control
        else:
            raise ValueError("Needs 1 or 4 arguments; received %i."
                             % len(create_from))

        if reduction_options:
            Nr, self.A, self.B, self.C, self.hsv = \
                self.reduction_functions[reduction_options.pop("reduction")](
                    self.A, self.B, self.C, **reduction_options
                    )

        self.state = None
        self.order = self.A.shape[0]

    @property
    def x(self):
        """State of the system at current time `t`

        """
        try:
            x = self.state.y
        except AttributeError:
            x = self.x0
        return x

    @property
    def t(self):
        """Current time of the system

        """
        return self.state.t

    @property
    def y(self):
        if callable(self.control):
            u = self.control
        else:
            def u(t, y):
                return self.control
        return np.dot(self.C, self.x) + np.dot(self.D, u(t, self.x))

    def f(self, t, y, u):
        """Rhs of the differential equation

        """
        if callable(u):
            u = u(t, y)
        else:
            u = np.asarray(u)
        return np.dot(self.A, y) + np.dot(self.B, u)

    def setupODE(self):
        """Set the ode solver. All integrator, options and initial value can 
        be set through class attributes.

        """
        self.state = ode(self.f)
        self.state.set_integrator(self.integrator,**self.integrator_options)
        if self.x0 is None:
            self.x0 = np.zeros((self.order,))
        self.state.set_initial_value(self.x0, self.t0)
        self.state.set_f_params(self.control)

    def __call__(self, times, control=None, force_ode_reset=False):
        """Get the output at specified times with a provided control

        It is possible to only request the output at one particular time
        or provide a list of times. If `times` is a sequence, the output
        will be a list of ``nparrays`` at these times, otherwise it's just
        a single ``nparray``. However the control can either be specified as a
        function or is a constant array over all times.

        Parameters
        ----------
        times : list or scalar
            The output for these timese will be calculated
        control : callable ``control(t, y)`` or array_like, optional
            If it is specified, it will be overwritten in the attributes.
        force_ode_reset : Boolean, optional
            If it's called, the ode solver is reset and the current attributes
            are used.

        """

        if control is not None:
            self.control = control
            if self.state and not force_ode_reset:
                self.state.set_f_params(control)

        if force_ode_reset or not self.state:
            self.setupODE()

        if not isinstance(times, collections.Sequence):
            times = [times]
            return_list = False
        else:
            return_list = True

        results = []

        if callable(self.control):
            u = self.control
        else:
            def u(t, y):
                return self.control
        for t in times:
            self.solve(t)
            results.append(np.dot(self.C, self.x) +
                           np.dot(self.D, u(t, self.x)))

        return results if return_list else results[0]

    def solve(self, t):
        if not self.state:
            self.setupODE()
        self.state.integrate(t)
        return self.x

