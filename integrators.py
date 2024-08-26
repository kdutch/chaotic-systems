"""
Created on Thu Oct 28 10:59:07 2021

@author: Kade

RK45 integrator wth adaptive step-sizing.

"""
# -*- coding: utf-8 -*-
import logging
import numpy as np

from typing import Optional, Tuple, Union


class RK45Integrator:
    #defines k1-6 based on coefficients from formula 2, table 3, in Fhelberg
    H_COEFF = np.array([(1/4), (3/8), (12/13), (1/2)])
    K2_COEFF = np.array([1/4])
    K3_COEFF = np.array([3/32, 9/32])
    K4_COEFF = np.array([1932/2197, -7200/2197, 7296/2197])
    K5_COEFF = np.array([439/216, -8, 3680/513,  -845/4104])
    K6_COEFF = np.array([-8/27, 2, -3544/2565, 1859/4104, -11/40])
    Y_COEFF = np.array([25/216, 1408/2565, 2197/4104, -1/5]) 
    ERROR_COEFF = np.array([1/360, -128/4275, -2197/75240, 1/50, 2/55])
    
    def __init__(self, tol: float, max_steps: Optional[int] = None):
        """
        Parameters
        ----------
        tol : float
            DESCRIPTION.
        max_step : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._tol = tol
        self._max_steps = max_steps
        
    @property
    def tol(self) -> float:
        """

        Returns
        -------
        int
            The tolerance for this instance of the Integrator.

        """
        return self._tol
    
    @tol.setter
    def set_tolerance(self, new_tolerance: float):
        """
        Sets the new tolerance.
        
        Parameters
        ----------
        new_tolerance: float
            The new tolerance for this instance of the integrator.

        """
        assert new_tolerance > 0, logging.error("The tolerance must be "
                                                "greater than 0.")
        self._tol = new_tolerance
    
    @property
    def max_steps(self) -> int:
        """
        Returns the value of the max_steps
        
        Returns
        -------
        self.max_steps: int
            The max_steps as defined by the instance of the integrator.
            
        """
        return self._max_steps
        
    @max_steps.setter
    def set_max_step(self, max_steps: int = Optional[None]):
        """
        Sets the new max_step value.
        
        PARAMETERS
        ----------
        max_steps: int
            The new max-steps for the integrator. 
            If no value is provided, sets max_steps to the default (10000).
            
        """
        if max_steps:
            assert max_steps > 0
        else:
            # defaul max value is 10,000
            max_steps = 10000
        self._max_step = max_steps
        
    def integrate(self, f: callable, t0: float, t1: float, y0: float, 
                  hmax: float, hmin: float, *args ,**kwargs) -> Tuple[np.array, np.array, np.array]:
        """
        Integrates the function with the provided values.
        
        Parameters
        ----------
        f : callable
            The function to integrate.
        t0 : float
            inital time. Where the integration begins.
        t1 : float
            boundary time. Where we are expected to stop the integration, 
            if we have not exceeded the max step-size.
        y0 : float
            Initial value of the function we are integrating.
        tol : float
            The tolerance for error when integrating.
        hmax : float
            Maximum step-size.
        hmin : float
            Minimum step-size.
        maxstep : int
            The maximum number of steps to take.
        X : np.array
            An array of values containing the values we are synchronizing to.
        *args : tuple
            Arguments to submit to f
        **kwargs : Dict
            Key-word arguments to submit to f.

        Returns
        -------
        (y, t, h) : Tuple(np.array[], np.array[], np.array[])
            Returns a tuple of arrays containing y (the result of the 
            integration), t, the axist against which we integrated (time), 
            and h, the step-sizes for each step of the integration.
            
        """
        step = 0
        # constrain the amount of memory that's allowed to be used by defining
        # a numpy array zeros of max size
        t, y, h_steps = (np.zeros(self.max_steps),
                         np.zeros((self.max_steps, y0.size)), 
                         np.zeros(self.max_steps))
        t[0], y[0] = t0, y0
        #guess first h
        h, h_steps[0] = hmax, hmax
        h = hmax
        h_steps[0] = h
        # run through the integrator a maxmimum of maxstep times,
        # and ensures we haven't hit our boundary condition
        while (t[step] < t1 and step < self.max_steps - 1): 
            # check to see if we're reached t1
            if ((t[step] + h) > t1):
                # if we have, make last step just large enough to get us to t1
                h = t1 - t[step] 
            # call the next solve next rk45step
            y[step+1], t[step+1], h = self.step(f, t[step], y[step], h, hmax, 
                                                hmin, self.tol, *args, **kwargs)
            h_steps[step+1] = h
            step +=1 
        # truncate the array to the number of steps needed to get to t1 
        # (or maxsteps, whichever is smaller)
        return y[0:step+1], t[0:step+1], h_steps[0:step+1] 
            
    @staticmethod
    def step(f: callable, t0: float, y0: float, h: float, hmax: float,
             hmin: float, tol: float, *args, **kwargs) -> Tuple[Union[float, np.array], float, float]:
        """
        A single RK45 integrator step. Pioritizes efficiency over 
        resolution/accuracy. If our resolution < pre-defined tolerance, 
        but the if the step, h, was already at minimum, the function decides 
        to continue to the next step anyway, even if it is outside the given
        tolerance.
        
        Parameters
        ----------
        f : callable
            The function to incrementally integrate.
        t0 : float
            Inital time.
        y0 : float
            Initial value.
        h : float
            Step-size.
        hmax : float
            Maximum step-size.
        hmin : floar
            Minimum step-size.
        tol : float
            Tolerance.
        *args : Tuple
            Any function arguments.
        **kwargs: Dict
            Any function key-word arguments.
            
        Returns
        -------
        y : float
            Current value (result of the inegration).
        t : float
            Current time value.
        h_new : float
            New step-size.

        """
        rk45 = RK45Integrator
        k1 = h*f(t0, y0, *args, **kwargs)
        k2 = h*f(t0 + rk45.H_COEFF[0]*h,   y0 + rk45.K2_COEFF[0]*k1, 
                                                *args, **kwargs)
        k3 = h*f(t0 + rk45.H_COEFF[1]*h,   y0 + rk45.K3_COEFF[0]*k1 + 
                                                rk45.K3_COEFF[1]*k2, 
                                                *args, **kwargs)
        k4 = h*f(t0 + rk45.H_COEFF[2]*h,   y0 + rk45.K4_COEFF[0]*k1 + 
                                                rk45.K4_COEFF[1]*k2 + 
                                                rk45.K4_COEFF[2]*k3, 
                                                *args, **kwargs)
        k5 = h*f(t0 + h,                   y0 + rk45.K5_COEFF[0]*k1 + 
                                                rk45.K5_COEFF[1]*k2 + 
                                                rk45.K5_COEFF[2]*k3 + 
                                                rk45.K5_COEFF[3]*k4, 
                                                *args, **kwargs)
        k6 = h*f(t0 + rk45.H_COEFF[3]*h,   y0 + rk45.K6_COEFF[0]*k1 + 
                                                rk45.K6_COEFF[1]*k2 + 
                                                rk45.K6_COEFF[2]*k3 + 
                                                rk45.K6_COEFF[3]*k4 + 
                                                rk45.K6_COEFF[4]*k5,
                                                *args, **kwargs)
        # defines y
        y = y0 + (rk45.Y_COEFF[0]*k1 + 
                  rk45.Y_COEFF[1]*k3 + 
                  rk45.Y_COEFF[2]*k4 + 
                  rk45.Y_COEFF[3]*k5)
        # computes the magnitude of the error
        r = (1/h)*np.sqrt(np.sum((rk45.ERROR_COEFF[0]*k1 + 
                                  rk45.ERROR_COEFF[1]*k3 + 
                                  rk45.ERROR_COEFF[2]*k4 +
                                  rk45.ERROR_COEFF[3]*k5 +
                                  rk45.ERROR_COEFF[4]*k6)**2))
        # if r exceeds instrument precision, set it to a small value 
        # will happen if eq approaches 0)
        if (hmin == hmax):
            t = t0 + hmin
            h_new = hmin
        else:
            if (r == 0):
                r = 2**(-15)
            s = (tol/(2*r))**(1/4)
            # defines new step-size
            h_new = s*h
            # if h_new > h_max, set h_new to h_max
            if (h_new > hmax):
                h_new = hmax
            # else-if h_new < h_min, set h_new to h_min
            elif (h_new < hmin):
                h_new = hmin
            # check to make sure the step was within the given tolerance    
            if (r <= tol):
                t = t0 + h
            # if not, repeat the step with the new step_size
            else:
                # if the step isn't at minimumm, repeats the step with
                # smaller step size
                if (h != hmin):
                    y, t, h_new = rk45.step(f, t0, y0, h_new, hmax, hmin, tol,
                                            *args, **kwargs)
                else:
                    t = t0 + h
            # after this is done, return the new values   
        return y, t, h_new



class SynchronizedRK45(RK45Integrator):
    
    def synchronize(self, f: callable, t0: float, t1: float, y0: float, 
                    hmax: float, hmin: float,  X: np.array, *args ,**kwargs):
        """
        Modified version of the integrate method that takes in an additional
        parameter and submits it to the funtion. This parameter is an array
        containing the result of a previous integration we are attempting to
        synchronize to.
        
        Parameters
        ----------
        f : callable
            The function to integrate.
        t0 : float
            inital time. Where the integration begins.
        t1 : float
            boundary time. Where we are expected to stop the integration, 
            if we have not exceeded the max step-size.
        y0 : float
            Initial value of the function we are integrating.
        tol : float
            The tolerance for error when integrating.
        hmax : float
            Maximum step-size.
        hmin : float
            Minimum step-size.
        maxstep : int
            The maximum number of steps to take.
        X : np.array
            An array of values containing the values we are synchronizing to.
        *args : tuple
            Arguments to submit to f
        **kwargs : Dict
            Key-word arguments to submit to f.

        Returns
        -------
        (y, t, h) : Tuple(np.array[], np.array[], np.array[])
            Returns a tuple of arrays containing y (the result of the 
            integration), t, the axist against which we integrated (time), 
            and h, the step-sizes for each step of the integration.

        """
        step = 0
        # constrain the amount of memory that's allowed to be used by
        # defining a numpy array of zeroes of max size
        t, y, h_steps = (np.zeros(self.max_steps), 
                         np.zeros((self.max_steps, y0.size)), 
                         np.zeros(self.max_steps))
        t[0], y[0] = t0, y0
        #guess first h
        h, h_steps[0] = hmax, hmax
        # run through the integrator a maxmimum of maxstep times
        # check to make sure we haven't exceeded t1 or maxsteps
        while (t[step] < t1 and step < self.max_steps - 1 and step < len(X)):
            # check to see if we're reached t1
            if ((t[step] + h) > t1):
                # if we have, make last step just large enough to get us to t1
                h = t1 - t[step] 
            # call the next solve next rk45step
            y[step+1], t[step+1], h = self.step(f, t[step], y[step],
                                                h, hmax, hmin, self.tol, 
                                                X[step], *args, **kwargs)
            h_steps[step+1] = h
            step +=1 
        # truncate the array to the number of steps needed to get to t1 
        # (or maxsteps, whichever is smaller)
        return y[0:step+1], t[0:step+1], h_steps[0:step+1] 
    