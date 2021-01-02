# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, bad-whitespace
# pylint: disable=too-many-lines, too-many-locals, len-as-condition
# pylint: disable=import-outside-toplevel

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import absolute_import, division, unicode_literals

import math
from math import cos, sin
import random
import warnings
import numpy as np
from numpy.linalg import inv
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spln
from scipy.stats import norm, multivariate_normal



# Older versions of scipy do not support the allow_singular keyword. I could
# check the version number explicily, but perhaps this is clearer
_support_singular = True
try:
    multivariate_normal.logpdf(1, 1, 1, allow_singular=True)
except TypeError:
    warnings.warn(
        'You are using a version of SciPy that does not support the '\
        'allow_singular parameter in scipy.stats.multivariate_normal.logpdf(). '\
        'Future versions of FilterPy will require a version of SciPy that '\
        'implements this keyword',
        DeprecationWarning)
    _support_singular = False


def _validate_vector(u, dtype=None):
    # this is taken from scipy.spatial.distance. Internal function, so
    # redefining here.

    u = np.asarray(u, dtype=dtype).squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def mahalanobis(x, mean, cov):
    """
    Computes the Mahalanobis distance between the state vector x from the
    Gaussian `mean` with covariance `cov`. This can be thought as the number
    of standard deviations x is from the mean, i.e. a return value of 3 means
    x is 3 std from mean.

    Parameters
    ----------
    x : (N,) array_like, or float
        Input state vector

    mean : (N,) array_like, or float
        mean of multivariate Gaussian

    cov : (N, N) array_like  or float
        covariance of the multivariate Gaussian

    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `x` and `mean`

    Examples
    --------
    >>> mahalanobis(x=3., mean=3.5, cov=4.**2) # univariate case
    0.125

    >>> mahalanobis(x=3., mean=6, cov=1) # univariate, 3 std away
    3.0

    >>> mahalanobis([1., 2], [1.1, 3.5], [[1., .1],[.1, 13]])
    0.42533327058913922
    """

    x = _validate_vector(x)
    mean = _validate_vector(mean)

    if x.shape != mean.shape:
        raise ValueError("length of input vectors must be the same")

    y = x - mean
    S = np.atleast_2d(cov)

    dist = float(np.dot(np.dot(y.T, inv(S)), y))
    return math.sqrt(dist)


def log_likelihood(z, x, P, H, R):
    """
    Returns log-likelihood of the measurement z given the Gaussian
    posterior (x, P) using measurement function H and measurement
    covariance error R
    """
    S = np.dot(H, np.dot(P, H.T)) + R
    return logpdf(z, np.dot(H, x), S)


def likelihood(z, x, P, H, R):
    """
    Returns likelihood of the measurement z given the Gaussian
    posterior (x, P) using measurement function H and measurement
    covariance error R
    """
    return np.exp(log_likelihood(z, x, P, H, R))


def logpdf(x, mean=None, cov=1, allow_singular=True):
    """
    Computes the log of the probability density function of the normal
    N(mean, cov) for the data x. The normal may be univariate or multivariate.

    Wrapper for older versions of scipy.multivariate_normal.logpdf which
    don't support support the allow_singular keyword prior to verion 0.15.0.

    If it is not supported, and cov is singular or not PSD you may get
    an exception.

    `x` and `mean` may be column vectors, row vectors, or lists.
    """

    if mean is not None:
        flat_mean = np.asarray(mean).flatten()
    else:
        flat_mean = None

    flat_x = np.asarray(x).flatten()

    if _support_singular:
        return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)
    return multivariate_normal.logpdf(flat_x, flat_mean, cov)


def gaussian(x, mean, var, normed=True):
    """
    returns probability density function (pdf) for x given a Gaussian with the
    specified mean and variance. All must be scalars.

    gaussian (1,2,3) is equivalent to scipy.stats.norm(2, math.sqrt(3)).pdf(1)
    It is quite a bit faster albeit much less flexible than the latter.

    Parameters
    ----------

    x : scalar or array-like
        The value(s) for which we compute the distribution

    mean : scalar
        Mean of the Gaussian

    var : scalar
        Variance of the Gaussian

    normed : bool, default True
        Normalize the output if the input is an array of values.

    Returns
    -------

    pdf : float
        probability distribution of x for the Gaussian (mean, var). E.g. 0.101 denotes
        10.1%.

    Examples
    --------

    >>> gaussian(8, 1, 2)
    1.3498566943461957e-06

    >>> gaussian([8, 7, 9], 1, 2)
    array([1.34985669e-06, 3.48132630e-05, 3.17455867e-08])
    """

    pdf = ((2*math.pi*var)**-.5) * np.exp((-0.5*(np.asarray(x)-mean)**2.) / var)
    if normed and len(np.shape(pdf)) > 0:
        pdf = pdf / sum(pdf)

    return pdf



def mul(mean1, var1, mean2, var2):
    """
    Multiply Gaussian (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean, var).

    Strictly speaking the product of two Gaussian PDFs is a Gaussian
    function, not Gaussian PDF. It is, however, proportional to a Gaussian
    PDF, so it is safe to treat the output as a PDF for any filter using
    Bayes equation, which normalizes the result anyway.

    Parameters
    ----------
    mean1 : scalar
         mean of first Gaussian

    var1 : scalar
         variance of first Gaussian

    mean2 : scalar
         mean of second Gaussian

    var2 : scalar
         variance of second Gaussian

    Returns
    -------
    mean : scalar
        mean of product

    var : scalar
        variance of product

    Examples
    --------
    >>> mul(1, 2, 3, 4)
    (1.6666666666666667, 1.3333333333333333)

    References
    ----------
    Bromily. "Products and Convolutions of Gaussian Probability Functions",
    Tina Memo No. 2003-003.
    http://www.tina-vision.net/docs/memos/2003-003.pdf
    """

    mean = (var1*mean2 + var2*mean1) / (var1 + var2)
    var = 1 / (1/var1 + 1/var2)
    return (mean, var)


def mul_pdf(mean1, var1, mean2, var2):
    """
    Multiply Gaussian (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean, var, scale_factor).

    Strictly speaking the product of two Gaussian PDFs is a Gaussian
    function, not Gaussian PDF. It is, however, proportional to a Gaussian
    PDF. `scale_factor` provides this proportionality constant

    Parameters
    ----------
    mean1 : scalar
         mean of first Gaussian

    var1 : scalar
         variance of first Gaussian

    mean2 : scalar
         mean of second Gaussian

    var2 : scalar
         variance of second Gaussian

    Returns
    -------
    mean : scalar
        mean of product

    var : scalar
        variance of product

    scale_factor : scalar
        proportionality constant


    Examples
    --------
    >>> mul(1, 2, 3, 4)
    (1.6666666666666667, 1.3333333333333333)

    References
    ----------
    Bromily. "Products and Convolutions of Gaussian Probability Functions",
    Tina Memo No. 2003-003.
    http://www.tina-vision.net/docs/memos/2003-003.pdf
    """

    mean = (var1*mean2 + var2*mean1) / (var1 + var2)
    var = 1. / (1./var1 + 1./var2)

    S = math.exp(-(mean1 - mean2)**2 / (2*(var1 + var2))) / \
                 math.sqrt(2 * math.pi * (var1 + var2))

    return mean, var, S


def add(mean1, var1, mean2, var2):
    """
    Add the Gaussians (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean,var).

    var1 and var2 are variances - sigma squared in the usual parlance.
    """

    return (mean1+mean2, var1+var2)


def multivariate_gaussian(x, mu, cov):
    """
    This is designed to replace scipy.stats.multivariate_normal
    which is not available before version 0.14. You may either pass in a
    multivariate set of data:

    .. code-block:: Python

       multivariate_gaussian (array([1,1]), array([3,4]), eye(2)*1.4)
       multivariate_gaussian (array([1,1,1]), array([3,4,5]), 1.4)

    or unidimensional data:

    .. code-block:: Python

       multivariate_gaussian(1, 3, 1.4)

    In the multivariate case if cov is a scalar it is interpreted as eye(n)*cov

    The function gaussian() implements the 1D (univariate)case, and is much
    faster than this function.

    equivalent calls:

    .. code-block:: Python

      multivariate_gaussian(1, 2, 3)
      scipy.stats.multivariate_normal(2,3).pdf(1)


    Parameters
    ----------

    x : float, or np.array-like
       Value to compute the probability for. May be a scalar if univariate,
       or any type that can be converted to an np.array (list, tuple, etc).
       np.array is best for speed.

    mu :  float, or np.array-like
       mean for the Gaussian . May be a scalar if univariate,  or any type
       that can be converted to an np.array (list, tuple, etc).np.array is
       best for speed.

    cov :  float, or np.array-like
       Covariance for the Gaussian . May be a scalar if univariate,  or any
       type that can be converted to an np.array (list, tuple, etc).np.array is
       best for speed.

    Returns
    -------

    probability : float
        probability for x for the Gaussian (mu,cov)
    """

    warnings.warn(
        ("This was implemented before SciPy version 0.14, which implemented "
         "scipy.stats.multivariate_normal. This function will be removed in "
         "a future release of FilterPy"), DeprecationWarning)

    # force all to numpy.array type, and flatten in case they are vectors
    x = np.array(x, copy=False, ndmin=1).flatten()
    mu = np.array(mu, copy=False, ndmin=1).flatten()

    nx = len(mu)
    cov = _to_cov(cov, nx)


    norm_coeff = nx*math.log(2*math.pi) + np.linalg.slogdet(cov)[1]

    err = x - mu
    if sp.issparse(cov):
        numerator = spln.spsolve(cov, err).T.dot(err)
    else:
        numerator = np.linalg.solve(cov, err).T.dot(err)

    return math.exp(-0.5*(norm_coeff + numerator))


def multivariate_multiply(m1, c1, m2, c2):
    """
    Multiplies the two multivariate Gaussians together and returns the
    results as the tuple (mean, covariance).

    Examples
    --------

    .. code-block:: Python

        m, c = multivariate_multiply([7.0, 2], [[1.0, 2.0], [2.0, 1.0]],
                                     [3.2, 0], [[8.0, 1.1], [1.1,8.0]])

    Parameters
    ----------

    m1 : array-like
        Mean of first Gaussian. Must be convertable to an 1D array via
        numpy.asarray(), For example 6, [6], [6, 5], np.array([3, 4, 5, 6])
        are all valid.

    c1 : matrix-like
        Covariance of first Gaussian. Must be convertable to an 2D array via
        numpy.asarray().

     m2 : array-like
        Mean of second Gaussian. Must be convertable to an 1D array via
        numpy.asarray(), For example 6, [6], [6, 5], np.array([3, 4, 5, 6])
        are all valid.

    c2 : matrix-like
        Covariance of second Gaussian. Must be convertable to an 2D array via
        numpy.asarray().

    Returns
    -------

    m : ndarray
        mean of the result

    c : ndarray
        covariance of the result
    """

    C1 = np.asarray(c1)
    C2 = np.asarray(c2)
    M1 = np.asarray(m1)
    M2 = np.asarray(m2)

    sum_inv = np.linalg.inv(C1+C2)
    C3 = np.dot(C1, sum_inv).dot(C2)

    M3 = (np.dot(C2, sum_inv).dot(M1) +
          np.dot(C1, sum_inv).dot(M2))

    return M3, C3


def covariance_ellipse(P, deviations=1):
    """
    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters
    ----------

    P : nd.array shape (2,2)
       covariance matrix

    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.

    Returns (angle_radians, width_radius, height_radius)
    """

    U, s, _ = linalg.svd(P)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width = deviations * math.sqrt(s[0])
    height = deviations * math.sqrt(s[1])

    if height > width:
        raise ValueError('width must be greater than height')

    return (orientation, width, height)


def _eigsorted(cov, asc=True):
    """
    Computes eigenvalues and eigenvectors of a covariance matrix and returns
    them sorted by eigenvalue.

    Parameters
    ----------
    cov : ndarray
        covariance matrix

    asc : bool, default=True
        determines whether we are sorted smallest to largest (asc=True),
        or largest to smallest (asc=False)

    Returns
    -------
    eigval : 1D ndarray
        eigenvalues of covariance ordered largest to smallest

    eigvec : 2D ndarray
        eigenvectors of covariance matrix ordered to match `eigval` ordering.
        I.e eigvec[:, 0] is the rotation vector for eigval[0]
    """

    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()
    if not asc:
        # sort largest to smallest
        order = order[::-1]

    return eigval[order], eigvec[:, order]


def _std_tuple_of(var=None, std=None, interval=None):
    """
    Convienence function for plotting. Given one of var, standard
    deviation, or interval, return the std. Any of the three can be an
    iterable list.

    Examples
    --------
    >>>_std_tuple_of(var=[1, 3, 9])
    (1, 2, 3)

    """

    if std is not None:
        if np.isscalar(std):
            std = (std,)
        return std


    if interval is not None:
        if np.isscalar(interval):
            interval = (interval,)

        return norm.interval(interval)[1]

    if var is None:
        raise ValueError("no inputs were provided")

    if np.isscalar(var):
        var = (var,)
    return np.sqrt(var)


def norm_cdf(x_range, mu, var=1, std=None):
    """
    Computes the probability that a Gaussian distribution lies
    within a range of values.

    Parameters
    ----------

    x_range : (float, float)
        tuple of range to compute probability for

    mu : float
        mean of the Gaussian

    var : float, optional
        variance of the Gaussian. Ignored if `std` is provided

    std : float, optional
       standard deviation of the Gaussian. This overrides the `var` parameter

    Returns
    -------

    probability : float
        probability that Gaussian is within x_range. E.g. .1 means 10%.
    """

    if std is None:
        std = math.sqrt(var)
    return abs(norm.cdf(x_range[0], loc=mu, scale=std) -
               norm.cdf(x_range[1], loc=mu, scale=std))


def _to_cov(x, n):
    """
    If x is a scalar, returns a covariance matrix generated from it
    as the identity matrix multiplied by x. The dimension will be nxn.
    If x is already a 2D numpy array then it is returned unchanged.

    Raises ValueError if not positive definite
    """

    if np.isscalar(x):
        if x < 0:
            raise ValueError('covariance must be > 0')
        return np.eye(n) * x

    x = np.atleast_2d(x)
    try:
        # quickly find out if we are positive definite
        np.linalg.cholesky(x)
    except:
        raise ValueError('covariance must be positive definit')

    return x


def rand_student_t(df, mu=0, std=1):
    """
    return random number distributed by student's t distribution with
    `df` degrees of freedom with the specified mean and standard deviation.
    """

    x = random.gauss(0, std)
    y = 2.0*random.gammavariate(0.5 * df, 2.0)
    return x / (math.sqrt(y / df)) + mu


def NEES(xs, est_xs, ps):
    """
    Computes the normalized estimated error squared (NEES) test on a sequence
    of estimates. The estimates are optimal if the mean error is zero and
    the covariance matches the Kalman filter's covariance. If this holds,
    then the mean of the NEES should be equal to or less than the dimension
    of x.

    Examples
    --------

    .. code-block: Python

        xs = ground_truth()
        est_xs, ps, _, _ = kf.batch_filter(zs)
        NEES(xs, est_xs, ps)

    Parameters
    ----------

    xs : list-like
        sequence of true values for the state x

    est_xs : list-like
        sequence of estimates from an estimator (such as Kalman filter)

    ps : list-like
        sequence of covariance matrices from the estimator

    Returns
    -------

    errs : list of floats
       list of NEES computed for each estimate

    """

    est_err = xs - est_xs
    errs = []
    for x, p in zip(est_err, ps):
        errs.append(np.dot(x.T, linalg.inv(p)).dot(x))
    return errs
