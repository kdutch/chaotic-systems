# -*- coding: utf-8 -*-
"""
Contains the choatic systems being integrated.


@author: Kade
"""
import numpy as np


def lorenz(t, y0, sig, r, b):
    """
    The function that represents the chaotic Lrenz system of differential
    equations.
    
    Parameters
    ----------
    t : current time
        The axis we are measuring against
    y0 : np.float64
        Initial value (value of the derivative at this point)
    sig : np.float64
        coefficient (proportional to the Prandtl number)
    r : TYPE
        coefficient (proportional to the Rayleigh number.
    b : TYPE
        coefficient (related to certain properties of the system itself).

    Returns
    -------
    dydxdz : np.array
        the value of the differential equation at that point.

    """
    x = y0[0]
    y = y0[1]
    z = y0[2]
    dx = sig*(y-x)
    dy = r*x - y - x*z
    dz = x*y - b*z
    dydxdz = np.array([dx, dy, dz])
    return dydxdz


def reciever(X, t, y0, sig, r, b):
    """
    A modified lorenz system designed to synchronize with an incoming 
    "signal" (a previously integrated lorenz system).
    
    Parameters
    ----------
    X : The "signal" the function is synchronizing, at the step current step.

    Returns
    -------
    dydxdz : np.array
        the value of the differential equation at that point.

    """
    u = y0[0]
    v = y0[1]
    w = y0[2]
    du = sig*(v-u)
    dv = r*X - v - X*w
    dw = X*v - b*w
    dudvdw = np.array([du, dv, dw])
    return dudvdw
