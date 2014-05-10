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
        Nr, Ar, Br, Cr, hsv = ab09ad(dico,job,equil,n,m,p,
                                     self.A,self.B,self.C,nr,tol)
        self.hsv = hsv
   
        return StateSpaceSystem(Ar, Br, Cr, self.D)

class lss(object):
    """linear time independent state space system.

    Default contstructor is called like lss(A,B,C,D) and a system can
    easily be copied by calling lss(sys) where sys is a
    lss object itself.

    Parameters
    ----------
    A, B, C, D : array_like
        State-Space matrices. If one of the matrices is None, it is
        replaced by a zero matrix with appropriate dimensions.
    reduction : {'truncation_square_root'}, optional
        Choose method of reduction. If it isn't provided, matrices are 
        used without reduction.
    **reduction_options : dict, optional
        The arguments with which the reduction method is called.

    Attributes 
    ----------
    x0 : array_like, optional
        Initial state. Defaults to the zero state.
    t0 : float
        Initial value for `t`
    integrator : str
        Name of the integrator used by ``scipy.integrate.ode``
    integrator_options : 
        Options for the specified integrator that can be set.

    """

    x0 = None
    t0 = 0.0
    integrator = 'dopri5'
    integrator_options = {}

    def __init__(self, *create_from, reduction=None, **reduction_options):
        """Initialize a linear state space system

        """

        if len(create_from) == 4:
            (self.A, self.B, self.C, self.D) = abcd_normalize(*create_from)
        elif len(create_from) == 1:
            self.A = create_from[0].A
            self.B = create_from[0].B
            self.C = create_from[0].C
            self.D = create_from[0].D
        else:
            raise ValueError("Needs 1 or 4 arguments; received %i."
                             % len(create_from))

        if reduction:
            reduction_functions = {
                'truncation_square_root' : truncation_square_root
                }
            Nr, self.A, self.B, self.C, self.hsv = \
                reduction_functions[reduction](**reduction_options)


        self.state = None
        self.order = self.A.shape[0]
        self.inputs = self.B.shape[1]
        self.outputs = self.C.shape[0]
        self.control = np.zeros((self.inputs,))

    @property
    def x(self):
        """State of the system at current time `t`

        """
        return self.state.y

    @property
    def t(self):
        """Current time of the system

        """
        return self.state.t

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
        will be a list of `nparray`s at these times, otherwise it's just
        a single `nparray`. However the control can either be specified as a
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

def truncation_square_root(A, B, C, k=0, tol=0.0, balance=True, scale=True):
    """Perform truncation of a system. Scaling and balancing are optional

    This allows to reduce a linear state space system by either specifying it
    to a certain number of states `k` or by specifying a error tolerance `tol`
    relative to the input. In theory the most accurate results are achieved by
    using `balance` and `scale` but the size of the error strongly depends on
    the particular problem and scaling and balancing may in some cases cost
    too much.

    Parameters
    ----------
    A, B, C : array_like
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
    hsv : 
        Hankel singular values of the original system. The size of the error
        may be calculated based on this.

    Raises
    ------
    ValueError
        If the system that's provided is not stable (i.e. `A` has eigenvalues
        which have non-negative real parts)
    ControlSlycot
        If the slycot subroutine `ab09ad` can't be found. Occurs if the
        slycot package is not installed.

    """
    if not isStable(A):
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
    n = np.size(A,0)
    m = np.size(B,1)
    p = np.size(C,0)
    nr = k
    Nr, Ar, Br, Cr, hsv = ab09ad(dico,job,equil,n,m,p,
                                 self.A,self.B,self.C,nr,tol)


def isStable(A):
    D, V = np.linalg.eig(A)
    return (D.real < 0).all()





