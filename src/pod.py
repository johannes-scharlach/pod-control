"""Model order reduction for linear state space systems can be done with
Proper Orthogonal Decomposition (POD) methods. Some of them can be simply
applied when creating a system.

"""

from __future__ import division, print_function

import collections
import numpy as np
from scipy import linalg
from scipy.integrate import ode
# from scipy.signal.ltisys import abcd_normalize
# currently buggy. Will be fixed in scipy v0.15
# instead the following can be used temporarily
from futurescipy import abcd_normalize


def truncation_square_root(A, B, C,
                           k=None, tol=0.0,
                           balance=True, scale=True,
                           check_stability=False,
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
    n = np.size(A, 0)
    m = np.size(B, 1)
    p = np.size(C, 0)
    nr = k

    if not length_cache_array:
        length_cache_array = 5*n**2 + n*m + n*p + 10*n + m*p

    return ab09ad(dico, job, equil,
                  n, m, p,
                  A, B, C,
                  nr, tol,
                  ldwork=length_cache_array)


def truncation_square_root_trans_matrix(A, B, C,
                                        k=None, tol=0.0,
                                        overwrite_a=False,
                                        balance=True, check_stability=False,
                                        length_cache_array=None):
    """Truncate the system and return transition matrices

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
    T : ndarray
        Transformation matrix
    Ti : ndarray
        Inverse of the transformation matrix

    Raises
    ------
    ValueError
        If the system that's provided is not stable (i.e. `A` has eigenvalues
        which have non-negative real parts)
    ImportError
        If the slycot subroutine `ab09ad` can't be found. Occurs if the
        slycot package is not installed.

    Notes
    -----
    see truncation_square_root
    """
    if check_stability and not isStable(A):
        raise ValueError("This doesn't seem to be a stable system!")

    A, T, sdim = linalg.schur(A, sort='lhp', overwrite_a=overwrite_a)

    if sdim < A.shape[0]:
        raise ValueError("This is not a stable system. \n" +
                         "The eigenvalues are not all in the left half of " +
                         "the complex plane.")

    TH = T.transpose().conj()
    B, C = np.dot(TH, B), np.dot(C, T)

    nr, A, B, C, hsv, T_, Ti_ = \
        truncation_square_root_schur(A, B, C,
                                     k=k, tol=tol,
                                     balance=balance,
                                     length_cache_array=length_cache_array)

    T, Ti = np.dot(T, T_), np.dot(Ti_, TH)

    return nr, A, B, C, hsv, T, Ti


def truncation_square_root_schur(A, B, C,
                                 k=None, tol=0.0,
                                 balance=True,
                                 length_cache_array=None):
    """Use balanced truncation on a system with A in real Schur form

    Notes
    -----
    see truncation_square_root_trans_matrix

    """
    try:
        from slycot import ab09ax
    except ImportError:
        raise ImportError("can't find slycot subroutine ab09ax")

    if balance:
        job = 'B'
    else:
        job = 'N'

    dico = 'C'
    n = np.size(A, 0)
    m = np.size(B, 1)
    p = np.size(C, 0)
    nr = k

    if not length_cache_array:
        length_cache_array = 5*n**2 + n*m + n*p + 10*n + m*p

    nr, A, B, C, hsv, T_, Ti_ = \
        ab09ax(dico, job,
               n, m, p,
               A, B, C,
               nr=nr, tol=tol,
               ldwork=length_cache_array)

    return nr, A, B, C, hsv, T_, Ti_


def inoptimal_truncation_square_root(A, B, C, k, check_stability=False):
    """Use scipy to perform balanced truncation

    Use scipy to perform balanced truncation on a linear state-space system.
    This method is the natural application of scipy and inoptimal performance
    wise compared to `truncation_square_root_trans_matrix`

    Notes
    -----
    see truncation_square_root_trans_matrix

    """
    if check_stability and not isStable(A):
        raise ValueError("This doesn't seem to be a stable system!")
    AH = A.transpose().conj()
    P = linalg.solve_lyapunov(A, -np.dot(B, B.transpose().conj()))
    Q = linalg.solve_lyapunov(AH, -np.dot(C.transpose().conj(), C))

    U = linalg.cholesky(P).transpose().conj()
    L = linalg.cholesky(Q)

    W, Sigma, V = linalg.svd(np.dot(U.transpose().conj(), L),
                             full_matrices=False,
                             overwrite_a=True, check_finite=False)

    W1 = W[:, :k]
    Sigma1 = Sigma[:k]
    V1 = V[:, :k]

    Sigma1_pow_neg_half = np.diag(Sigma1**-.5)

    T1 = np.dot(Sigma1_pow_neg_half,
                np.dot(V1.transpose().conj(), L.transpose().conj()))
    Ti1 = np.dot(np.dot(U, W1),
                 Sigma1_pow_neg_half)

    return k, np.dot(T1, np.dot(A, Ti1)), np.dot(T1, B), np.dot(C, Ti1), \
        Sigma, Ti1, T1


def controllability_truncation(A, B, C, k,
                               check_stability=False,
                               use_scipy=None,
                               compute_svd=False):
    """Truncate the system based on the controllability Gramian

    Solves the Lyapunov Equation for ``AP + PA^H + B B^H`` and computes the
    eigenvalues and eigenvectors of `P` which are used to truncate the system.

    Parameters
    ----------
    A, B, C : array_like
        State-Space matrices of the system that should be reduced
    k : int
        Order of the reduced system
    check_stability : boolean, optional
        Should be checked if A is stable and hence the controllability Gramian
        exists
    use_scipy : boolean, optional
        Compute the solution of the Lyapunov Equation through
        ``scipy.linalg.solve_lyapunov`` if `True`, through ``slycot.sb03md``
        if `False` and if the value of this parameter is `None`,
        ``slycot.sb03md`` is used if available. The latter should lead to
        better performance.

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
    Lambdak : ndarray
        Largest k eigenvalues of the Gramian.
    Uk : ndarray
        Transformation matrix
    UkH : ndarray
        Hermetian of the transformation matrix
    """
    if check_stability and not isStable(A):
        raise ValueError("This doesn't seem to be a stable system!")

    N = A.shape[0]

    try:
        from slycot import sb03md
    except ImportError:
        if use_scipy is False:
            raise ImportError("can't find slycot subroutine sb03md")
        else:
            use_scipy = True
    else:
        if use_scipy is None:
            use_scipy = False

    if use_scipy:
        P = linalg.solve_lyapunov(A, -np.dot(B, B.conj().transpose()))
    else:
        res = sb03md(N, -np.dot(B, B.conj().transpose()), A, np.eye(N), 'C')
        P, scale, sep, ferr, w = res
        if scale < 1.:
            e = "Handling scale<1. is not yet implemented."
            raise NotImplementedError(e)

    Lambda, U = linalg.eigh(P, check_finite=False,
                            overwrite_a=False, overwrite_b=False)

    if compute_svd:
        s = linalg.svd(P, compute_uv=False)
    else:
        s = Lambda[range(N-1, -1, -1)]

    Uk = U[:, range(N-1, N-k-1, -1)]

    UkH = Uk.conj().transpose()

    A = np.dot(UkH, np.dot(A, Uk))
    B = np.dot(UkH, B)
    C = np.dot(C, Uk)

    return k, A, B, C, s, Uk, UkH


def isStable(A):
    """Check if all eigenvalues are in the left half of the complex plane"""
    D, V = linalg.eigh(A)
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
    reduction_functions = {
        'truncation_square_root': truncation_square_root,
        'controllability_truncation': controllability_truncation,
        'truncation_square_root_trans_matrix':
        truncation_square_root_trans_matrix,
        'truncation_square_root_schur': truncation_square_root_schur,
        'inoptimal_truncation_square_root': inoptimal_truncation_square_root}

    def __init__(self, *create_from, **reduction_options):
        """Initialize a linear state space system

        """

        if len(create_from) == 4:
            (self.A, self.B, self.C, self.D) = abcd_normalize(*create_from)
            self.control = np.zeros((self.inputs,))
        elif len(create_from) == 1:
            self.A = create_from[0].A
            self.B = create_from[0].B
            self.C = create_from[0].C
            self.D = create_from[0].D
            self.control = create_from[0].control
            self.x0 = create_from[0].x0
            self.t0 = create_from[0].t0
            self.integrator = create_from[0].integrator
            self.integrator_options = create_from[0].integrator_options
            self.reduction_functions = create_from[0].reduction_functions
        else:
            raise ValueError("Needs 1 or 4 arguments; received %i."
                             % len(create_from))

        if reduction_options.get("reduction", False):
            reduction_output = \
                self.reduction_functions[reduction_options.pop("reduction")](
                    self.A, self.B, self.C, **reduction_options)

            Nr, self.A, self.B, self.C, self.hsv = reduction_output[:5]
            if len(reduction_output) == 7:
                self.T, self.Ti = reduction_output[-2:]
                if self.x0 is not None:
                    if (self.x0 == 0).all():
                        self.x0 = np.zeros((self.order,))
                    else:
                        self.x0 = np.dot(self.Ti, self.x0)

        self.state = None

    @property
    def order(self):
        """Order of the system."""
        return self.A.shape[0]

    @property
    def inputs(self):
        """Number of inputs of the system."""
        return self.B.shape[1]

    @property
    def outputs(self):
        """Number of outputs of the system."""
        return self.C.shape[0]

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
        try:
            t = self.state.t
        except AttributeError:
            t = self.t0
        return t

    @property
    def y(self):
        if callable(self.control):
            u = self.control
        else:
            def u(t, y):
                return self.control
        return np.dot(self.C, self.x) + np.dot(self.D, u(self.t, self.x))

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
        self.state = ode(self.f, jac=lambda t, y, *f_args: self.A)
        self.state.set_integrator(self.integrator, **self.integrator_options)
        if self.x0 is None:
            x0 = np.zeros((self.order,))
        else:
            x0 = self.x0
        self.state.set_initial_value(x0, self.t0)
        self.state.set_f_params(self.control)

    def __call__(self, times, force_ode_reset=False):
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
        force_ode_reset : Boolean, optional
            If it's called, the ode solver is reset and the current attributes
            are used.

        """
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
        """Solve the ODE system"""
        if t is not self.t:
            self.state.integrate(t)
        return self.x
