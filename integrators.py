"""
Created on Thu Oct 28 10:59:07 2021

@author: Kade
"""

# -*- coding: utf-8 -*-
import numpy as np

#f is going to by dxdydz, the function we're integrating
#y0 is going to be an array of the initial values
#h is ininitial suggested stepsize
#nstep is number of steps  
def rk45(f, t0, t1, y0, tol, hmax, hmin, maxstep, *args):
    step = 0
    #constrain the amount of memory that's allowed to be used by defining a numpy array
    #of max size
    t = np.zeros(maxstep)
    y = np.zeros((maxstep, y0.size))
    h_steps = np.zeros(maxstep)
    #set initial values
    t[0] = t0
    y[0] = y0
    #guess first h
    h = hmax
    h_steps[0] = h
    #run through the integrator a maxmimum of maxstep times
    while (t[step] < t1 and step+1 < maxstep): #check to make sure we haven't exceeded t1 or maxsteps
        step +=1 #increment step size
        if ((t[step-1] + h) > t1): #check to see if we're reached t1
            h = t1 - t[step-1] #if we have, make last step just large enough to get us to t1
        y[step], t[step], h = rk45step(f, t[step-1], y[step-1], h, hmax, hmin, tol, *args) #call the next solve next rk45step
        h_steps[step] = h
    return y[0:step+1], t[0:step+1], h_steps[0:step+1] #truncate the array to the number of steps needed to get to t1 (or maxsteps, whichever is smaller)

def rk45step(f, t0, y0, h, hmax, hmin, tol, *args):
    #defines k1-6 based on coefficients from formula 2, table 3, in Fhelberg
    k1 = h*f(t0, y0, *args)
    k2 = h*f(t0 + (1/4)*h,   y0 + (1/4)*k1, *args)
    k3 = h*f(t0 + (3/8)*h,   y0 + (3/32)*k1     + (9/32)*k2, *args)
    k4 = h*f(t0 + (12/13)*h, y0 + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3, *args)
    k5 = h*f(t0 + h,         y0 + (439/216)*k1   - 8*k2           + (3680/513)*k3  - (845/4104)*k4, *args)
    k6 = h*f(t0 + (1/2)*h,   y0 - (8/27)*k1      + 2*k2           - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5, *args)
    #defines y
    y = y0 + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5
    #computes the magnitude of the error
    r = (1/h)*np.sqrt(np.sum(((1/360)*k1 - (128/4275)*k3 - (2197/75240)*k4 + (1/50)*k5 + (2/55)*k6)**2))
    #if r exceeds instrument precision, set it to a small value (will happen if eq approaches 0)
    if (hmin == hmax):
        t = t0 + hmin
        h_new = hmin
    else:
        if (r == 0):
            r = 2**(-15)
        s = (tol/(2*r))**(1/4)
        #defines new step-size
        h_new = s*h
        #if h_new > h_max, set h_new to h_max
        if (h_new > hmax):
            h_new = hmax
        #else-if h_new < h_min, set h_new to h_min
        elif (h_new < hmin):
            h_new = hmin
        #check to make sure the step was within the given tolerance    
        if (r <= tol):
            t = t0 + h
        #if not, repeat the step with the new step_size
        else:
            #pioritizes efficiency over resolution/accuracy
            #if the step was already at minimum, decides to continue to the next step anyway,
            #even if it is outside the given tolerance
            #if the step isn't at minimumm, repeats the step with smaller step size
            if (h != hmin):
                y, t, h_new = rk45step(f, t0, y0, h_new, hmax, hmin, tol, *args)
            else:
                t = t0 + h
        #after this is done, return the new values   
    return y, t, h_new

    
