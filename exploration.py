# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:27:59 2021

@author: Kade
"""
import numpy as np
import integrators
from functions import lorenz
from utilities import plot2d, plot3d, plot_3

#defines values and runs the integrator
def main():
    #define the values
    sig = 10
    b = 8/3
    r_values = [28]
    tol= 10**(-6)
    t0 = 0
    t1 = 1000
    maxstep = 1000000
    y0 = np.array([10,10,10])
    hmax = 0.1
    hmin = 0.5*10**(-9)
    #run the integrator
    for r in r_values:
        y, t, h = integrators.rk45(lorenz, t0, t1, y0, tol, hmax, hmin, maxstep, sig, r, b)
        #plot the results
        #'''
        plot3d(y[:,0], y[:,1], y[:,2], r)
        #plot2d(t, y[:,0], y[:,1], r, 'x', 'y')
        #plot2d(t, y[:,1], y[:,2], r, 'y', 'z')
        #plot2d(t, y[:,0], y[:,2], r, 'x', 'z')
        #title = 'intermittent solution | r=' + str(r)
        #plot_3(t, y[:,0], y[:,1], y[:,2], title, 'x', 'y', 'z', 't')
        #'''
        n = len(y[:,2])
        #plot2d(t, y[0:n-1, 2], y[1:n, 2], r, 'z_n', 'z_n+1')]
        #for fixed-point solutions
        print('x=' + str(y[n-1, 0])[0:4] + '   y=' + str(y[n-1, 1])[0:4] + '   z=' + str(y[n-1, 2])[0:4])
        print('h_min=' + str(h[1:h.size-1].min()))
        print(1.5*10**(-3))
    pass


main()