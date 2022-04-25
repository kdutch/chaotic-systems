# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:19:27 2021

@author: Kade
"""
import numpy as np
#collection of difq to be used

#define the lorenz system of equations as dxdydz
#returns a numpy array of dx, dy, dz
def lorenz(t, y0, sig, r, b):
    x = y0[0]
    y = y0[1]
    z = y0[2]
    dx = sig*(y-x)
    dy = r*x - y - x*z
    dz = x*y - b*z
    dydxdz = np.array([dx, dy, dz])
    return dydxdz

def reciever(t, y0, sig, r, b, X):
    u = y0[0]
    v = y0[1]
    w = y0[2]
    du = sig*(v-u)
    dv = r*X - v - X*w
    dw = X*v - b*w
    dudvdw = np.array([du, dv, dw])
    return dudvdw
