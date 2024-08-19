# -*- coding: utf-8 -*-
"""
Reads input from CLI and plots the chaotic lorenz system for many values of r.


@author: Kade
"""
import argparse
import numpy as np

from typing import Dict, Union

from functions import lorenz
from integrators import RK45Integrator
from plotting_utilities import plot2d, plot2d_three_params, plot3d


def parse_input_arguments() -> argparse.ArgumentParser:
    """
    Creates an argparser and gets the selected/default values.

    Returns
    -------
    argparse.NameSpace
        A Namespace containing all parsed input arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig", action='store', default=10, 
                        required=False, type=float)
    parser.add_argument("--b", default=8/3, action='store',
                        required=False, type=float)
    parser.add_argument("--r_values", default=[1, 10, 28], nargs='*', 
                        action='store', required=False, type=float)
    parser.add_argument("--tol", default=10**(-6), action='store',
                        required=False, type=float)
    parser.add_argument("--t0", default=0, action='store',
                        required=False, type=float)
    parser.add_argument("--t1", default=100, action='store',
                        required=False, type=float)
    parser.add_argument("--y0", default=[10, 10, 10], action='store', 
                        nargs='*', required=False, type=float)
    parser.add_argument("--hmax", default=0.1, action='store', 
                        required=False, type=float)
    parser.add_argument("--hmin", default=0.5*10**(-9), action='store', 
                        required=False)
    parser.add_argument("--max_steps", default=1000000, action='store', 
                        required=False, type=float)
    return parser.parse_args()

def sanitize_intput_arguments(parsed_args: argparse.Namespace) -> \
    Dict[str, Union[int, list]]:
    """
    Takes the input arguments and santizes some of the and decomposes them 
    into a dictionary.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        A Namespace containing all of the arguments .

    Returns
    -------
    dict
        A dictionary containing arguments parsed and cassed from the run 
        command.

    """
    return {
        "t0": parsed_args.t0, 
        "t1": parsed_args.t1, 
        "y0": np.array([float(arg) for arg in parsed_args.y0]),
        "hmax": parsed_args.hmax, 
        "hmin": parsed_args.hmin, 
        "sig": parsed_args.sig, 
        "r_values": [float(arg) for arg in parsed_args.r_values],
        "b": parsed_args.b,
        "tol": parsed_args.tol,
        "max_steps": parsed_args.max_steps
        }
    
    
def print_results(y: np.array, t: np.array, h: np.array, r: int):
    """
    Plots the results in 3D space and 2D space, and also prints the last
    x/y/z coordinates (so that you can observe the values in the case of a 
                       convergent solution.)
    
    Parameters
    ----------
    y: np.array[3xstep]
        A 3-by-by-step array containing the x, y, z coordinates at for 
        the system at step[idx]
    t: np.array[1xstep]
        A 1xstep array containing the time t at for step[idx]
    h: np.arrax[1xstep]
        An array containing the stepsize for each step
    r: int
        The r-value for the system.
        
    """
    title = ' r=' + str(r)
    plot3d(y[:,0], y[:,1], y[:,2], r)
    plot2d(t, y[:,0], y[:,1], r, 'x', 'y')
    plot2d(t, y[:,1], y[:,2], r, 'y', 'z')
    plot2d(t, y[:,0], y[:,2], r, 'x', 'z')
    plot2d_three_params(t, y[:,0], y[:,1], y[:,2], title, 'x', 'y', 'z', 't')
    n = len(y[:,2])
    plot2d(t, y[0:n-1, 2], y[1:n, 2], r, 'z_n', 'z_n+1')
    print('x=' + str(y[n-1, 0])[0:4] + '   y=' + str(y[n-1, 1])[0:4] + 
          '   z=' + str(y[n-1, 2])[0:4])
    print('h_min=' + str(h[1:h.size-1].min()))

    
def main():
    parsed_args = parse_input_arguments()
    kwargs = sanitize_intput_arguments(parsed_args)
    integrator = RK45Integrator(tol=kwargs.pop("tol"), 
                                max_steps=kwargs.pop("max_steps"))
    # run the integrator for each value of r
    for r in kwargs.pop('r_values'):
        y, t, h = integrator.integrate(lorenz, **kwargs, r=r)
        print_results(y, t, h, r)
        
        
main()