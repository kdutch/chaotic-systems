# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:03:07 2021

@author: Kade
"""
import numpy as np
from integrators import RK45Integrator
from functions import lorenz, reciever
from utilities import plot3d, plot2d, plot, plot_3, plot_2, readWave, writeWave
import matplotlib.pyplot as plt


class SynchronizedRK45(RK45Integrator):
    
    def integrate(self, f: callable, t0: float, t1: float, y0: float, 
                  hmax: float, hmin: float, *args ,**kwargs):
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



def main():
    sig = 10
    b = 8/3
    r = 28
    tol= 10**(-6)
    y0 = np.array([10, 10, 10])
    
    
    #check for synchronization conditions
    '''
    t0 = 0
    t1 = 5 #time*k
    h = 2.5*10**(-5)
    hmax = h
    hmin = h
    maxstep = 5000000
    r_values = [28]# [0, 10, 28, 30, 50, 75, 99, 100, 150, 200, 250]
    #synchronize the system
    for r in r_values:
        res, t, h = integrators.rk45(lorenz, t0, t1, y0, tol, hmax, hmin, maxstep, sig, r, b)
        x = res[:,0]
        y = res[:,1]
        z = res[:,2]
        u, v, w, t2, h = synchronize(reciever, t0, t1, np.array([0,0,0]), tol, hmax, hmin, maxstep, sig, r, b, x)
        title ='r=' + str(r) + ' | h=' + str(hmin)
        plot_3(t, np.abs(x-u), np.abs(y-v), np.abs(z-w), title, '|x-u|', '|y-v|', '|z-w|', 't')
        #plot_3(t[200000::], np.abs(x-u)[200000::], np.abs(y-v)[200000::], np.abs(z-w)[200000::], title, '|x-u|', '|y-v|', '|z-w|', 't')
        #plot(t, np.abs(x-u), title, 't', '|x-u|')
        #plot(t, np.abs(y-v), title, 't',  '|y-v|')
        #plot(t, np.abs(z-w), title, 't', '|z-w|')
        
    #'''
    
    #try to transmit a binary signal
    #'''
    #lets say 0 < t < 10 for this set
    #binary_signal generation:
    h = 5*10**(-4) #this is about twice the step size  
    A = np.random.rand(1)*0.15
    t0 = 0
    t1 = 10
    t_p = 0.25
    hmax = h
    hmin = h
    pulse_width = t_p/h
    frame_num = int(t1/h)
    maxstep = frame_num
    binary_signal = np.zeros(frame_num)
    b0 = 0
    b1 = int(pulse_width)
    i = 0
    integrator = RK45Integrator(tol=tol, max_steps=maxstep)
    #create binary signal
    while (b1 != b0):
        binary_signal[b0:b1] = A*(-1)**i
        i += 1
        b0 = b1
        b1 = int((i+1)*pulse_width)
        if (b1 > frame_num):
            b1 = frame_num
    t = np.linspace(t0, t1, frame_num)
    res, t, h = integrator.integrate(lorenz, t0, t1, y0, tol, hmax, hmin, maxstep, sig, r, b)
    x = res[:,0]
    X_t = x + binary_signal
    u, v, w, t, h = integrator.synchronize(reciever, t0, t1, y0, tol, hmax, hmin, maxstep, X_t, sig, r, b)
    binary_sent = X_t - u
    #plot_2(t, x, u, 'original vs. synchronized signal', 'original x', 'synchronized u', 't')
    title = 'original signal vs. extracted signal | A=' + str(A[0:4])
    plot_2(t, binary_signal, binary_sent, title, 'original signal', 'extracted signal', 't')
    #'''
    
    
    
    
    '''
    #get signal info and convert to an integer
    signal_raw, n, sample_width, sample_rate, num_frames, num_channels = readWave('wav/song_of_storms.wav')
    sample_time = 1/sample_rate
    time = sample_time*num_frames
        
    #define the values
    r=28
    h = sample_time
    hmax = h
    hmin = h
    offset = 2.5
    maxstep = int(2*num_frames + (offset)/h) #there are 2 channels
    print(sample_time)
    t0 = -offset
    t1 = time*2 #there are 2 channels
    norm = 10**(-6)
    percent = 0.1
    signal = norm*signal_raw
    i = np.linspace(0, signal.size, n)
    title = 'original signal'
    plot(i, signal_raw, title, 'frame', 'byte (int)')
    
    #integrate first system
    res, t, h = integrators.rk45(lorenz, t0, t1, y0, tol, hmax, hmin, maxstep, sig, r, b)
    x = res[:,0]
    title = 'original distribution | r=' + str(r)
    plot(t, x, title, 't', 'x')
    
    #perturb x(t)
    X_t = np.zeros(t.size)
    index = np.where( t >= 0)
    X_t[index] = signal
    X_t = X_t + x
    title = 'perterbed distribution | r=' + str(r) + '| norm=' + str(percent) + '%'
    plot(t, X_t, title, 't', 'X(t)')
    
    #synchronize the system
    u, v, w, t, h = synchronize(reciever, t0, t1, y0, tol, hmax, hmin, maxstep, sig, r, b, X_t)
    
    #decode the signal
    signal_recieved = X_t - u
    signal_recieved = signal_recieved[index]
    signal_sent = signal_recieved/norm
    title = 'sent signal | r=' + str(r) + ' | norm=' + str(percent) + '%'
    plot(i, signal_sent, title, 'frame', 'byte (int)')
    signal_decoded = signal_sent.astype('int16').tobytes()
    num_frames = signal_sent.size
    file_string = 'sent/song_of_storms_01_r=' + str(r)[0:4] + '_h=' + str(hmax) + '_norm=' + str(norm) + '.wav'
    writeWave(file_string, signal_decoded, sample_width, sample_rate, num_frames)
    #'''
    pass

main()