# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http:\\github.com\rlabbe\filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import scipy.linalg as linalg
from numpy import dot, zeros, eye



def dot3(A,B,C):
    """ Returns the matrix multiplication of A*B*C"""
    return dot(A, dot(B,C))


class KalmanFilter(object):

    def __init__(self, dim_x, dim_z):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------
        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.
        """

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.G = 0                # control transistion matrx
        self.F = 0                # state transition matrix
        self.H = 0                # Measurement function
        self.R = eye(dim_z)       # state uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.residual = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def update(self, Z, R=None):
        """
        Add a new measurement (Z) to the kalman filter. If Z is None, nothing
        is changed.

        Parameters
        ----------
        Z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        """

        if Z is None:
            return

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        H = self.H
        P = self.P
        x = self.x

        # y = Z - Hx
        # error (residual) between measurement and prediction
        self.residual = Z - dot(H, x)

        # S = HPH' + R
        # project system uncertainty into measurement space
        S = dot3(H, P, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        K = dot3(P, H.T, linalg.inv(S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(K, self.residual)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(K, H)
        self.P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)

        self.S = S
        self.K = K


    def predict(self, u=0):
        """ Predict next position.
        Parameters
        ----------
        u : np.array
            Optional control vector. If non-zero, it is multiplied by G
            to create the control input into the system.
        """

        # x = Fx + Gu
        self.x = dot(self.F, self.x) + dot(self.G, u)

        # P = FPF' + Q
        self.P = dot3(self.F, self.P, self.F.T) + self.Q


    def batch_filter(self, Zs, Rs=None, update_first=False):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------
        Zs : list-like
            list of measurements at each time step `self.dt` Missing
            measurements must be represented by 'None'.

        Rs : list-like, optional
            optional list of values to use for the measurement error
            covariance; a value of None in any position will cause the filter
            to use `self.R` for that time step.

        update_first : bool, optional,
            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.

        Returns
        -------

        means: np.array((n,dim_x,1))
            array of the state for each time step. Each entry is an np.array.
            In other words `means[k,:]` is the state at step `k`.

        covariance: np.array((n,dim_x,dim_x))
            array of the covariances for each time step. In other words
            `covariance[k,:,:]` is the covariance at step `k`.
        """

        n = np.size(Zs,0)
        if Rs is None:
            Rs = [None]*n

        # mean estimates from Kalman Filter
        means = zeros((n,self.dim_x,1))

        # state covariances from Kalman Filter
        covariances = zeros((n,self.dim_x,self.dim_x))

        if update_first:
            for i,(z,r) in enumerate(zip(Zs,Rs)):
                self.update(z,r)
                means[i,:] = self.x
                covariances[i,:,:] = self.P
                self.predict()
        else:
            for i,(z,r) in enumerate(zip(Zs,Rs)):
                self.predict()
                self.update(z,r)

                means[i,:] = self.x
                covariances[i,:,:] = self.P

        return (means, covariances)


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.

        Parameters
        ----------
        u : np.array
            optional control input

        Returns
        -------
        (x, P)
            State vector and covariance array of the prediction.
        """

        x = dot(self.F, self.x) + dot(self.G, u)
        P = dot3(self.F, self.P, self.F.T) + self.Q
        return (x, P)


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self.H, self.x)


    def measurement_of_state(self, x):
        """ Helper function that converts a state into a measurement.

        Parameters
        ----------
        x : np.array
            kalman state vector

        Returns
        -------
        z : np.array
            measurement corresponding to the given state
        """
        return dot(self.H, x)



class ExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z):
        """ Extended Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------
        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.
        """

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.G = 0                # control transition matrix
        self.F = 0                # state transition matrix
        self.R = eye(dim_z)       # state uncertainty
        self.Q = eye(dim_x)       # process uncertainty
        self.residual = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def predict_update(self, z, HJabobian, Hx, u=0):
        """ Performs the predict/update innovation of the extended Kalman
        filter.

        Parameters
        ----------
        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.


        Hx : function
            function which takes a state variable and returns the measurement
            that would correspond to that state.

        u : np.array or scalar
            optional control vector input to the filter.
        """

        F = self.F
        G = self.G
        P = self.P
        Q = self.Q
        R = self.R
        x = self.x

        H = HJabobian(x)

        # predict step
        x = dot(F, x) + dot(G, u)
        P = dot3(F, P, F.T) + Q

        # update step
        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))

        self.x = x + dot(K, (z - Hx(x)))

        I_KH = self._I - dot(K, H)
        self.P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def update(self, z, HJabobian, Hx):
        """ Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------
        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.


        Hx : function
            function which takes a state variable and returns the measurement
            that would correspond to that state.
        """

        P = self.P
        R = self.R
        x = self.x

        H = HJabobian(x)

        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))

        self.x = x + dot(K, (z - Hx(x)))

        I_KH = self._I - dot(K, H)
        self.P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def predict(self, u=0):
        """ Predict next position.
        Parameters
        ----------
        u : np.array
            Optional control vector. If non-zero, it is multiplied by G
            to create the control input into the system.
        """

        self.x = dot(self.F, self.x) + dot(self.G, u)
        self.P = dot3(self.F, self.P, self.F.T) + self.Q