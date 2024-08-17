# -*- coding: utf-8 -*-
"""
Reads input from CLI and plots the chaotic lorenze system for many values of r.


@author: Kade
"""
import argparse
import numpy as np

from typing import List

from integrators import RK45Integrator
from functions import lorenz
from utilities import plot2d, plot3d, plot_3

# Default integration values
SIG: int = 10
B: float = 8/3
R_VALUES: List[int] = [28]
TOL= 10**(-6)
T0 = 0
T1 = 100
MAX_STEP = 1000000
Y0 = np.array([10,10,10])
HMAX = 0.1
HMIN = 0.5*10**(-9)


def parse_input_arguments() -> argparse.ArgumentParser:
    """
    Creates an argparser and gets the selected/default values.

    Returns
    -------
    parser: argparse.AgrgumentParser
        DESCRIPTION.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("sig", action='store', default=10)
    parser.add_argument("b", default=8/3)
    parser.add_argument("r_values", default=[1, 10, 28], nargs='+', action='store')
    parser.add_argument("tol", default=10**(-6), action='store')
    parser.add_argument("t0", default=0, action='store')
    parser.add_argument("t1", default=100, action='store')
    parser.add_argument("y0", default=[10, 10, 10], action='store', nargs='+')
    parser.add_argument("hmax", default=0.1, action='store')
    parser.add_argument("hmin", default=0.5*10**(-9), action='store')
    parser.parse_args()
    return parser

def digest_input_argument(parser: argparse.ArgumentParser) -> dict:
    """
    Creates an argparser and gets the selected/default values.

    Returns
    -------
    parser: argparse.AgrgumentParser
        DESCRIPTION.

    """
    
    return parser

    
    
def print_results(y, t, h, r):
    """
    Plots the results for
    
    """
    title = ' r=' + str(r)
    plot3d(y[:,0], y[:,1], y[:,2], r)
    plot2d(t, y[:,0], y[:,1], r, 'x', 'y')
    plot2d(t, y[:,1], y[:,2], r, 'y', 'z')
    plot2d(t, y[:,0], y[:,2], r, 'x', 'z')
    plot_3(t, y[:,0], y[:,1], y[:,2], title, 'x', 'y', 'z', 't')
    n = len(y[:,2])
    plot2d(t, y[0:n-1, 2], y[1:n, 2], r, 'z_n', 'z_n+1')
    print('x=' + str(y[n-1, 0])[0:4] + '   y=' + str(y[n-1, 1])[0:4] + 
          '   z=' + str(y[n-1, 2])[0:4])
    print('h_min=' + str(h[1:h.size-1].min()))

    
#defines values and runs the integrator
def main():
    parse_input_arguments()
    #define the values
    integrator = RK45Integrator(tol=tol, max_steps=maxstep)
    #run the integrator
    for r in r_values:
        y, t, h = integrator.integrate(lorenz, T0, T1, Y0, HMAZ, hmin, sig, r, b)
        #for fixed-point solutions
        plot_results(y, t, h, r)
        
    pass


main()