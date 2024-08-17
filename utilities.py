# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 9:58:3 2021

Contains plotting utilities.

@author: Kade
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import wave


def plot3d(x, y, z, r):
    """
    Plots x, y, z in 3D. Also lists the R values. t is denoted by color, by the colormap
    is cyclical so the color repets
    
    
    """
    #set up the figure
    fig = plt.figure(figsize = (10,10), dpi=80)
    #print(plt.cm.get_cmap('inferno').N)
    ax = plt.axes(projection='3d')
    # plots in 3D. For loop was necessary for adusting the color of the line
    for i in range(len(x)-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.twilight((i+1)%510))
    #set values and labels
    r_val = 'r= ' + str(r)
    ax.text(-20, 30, 30, r_val, fontsize='x-large')
    ax.set_xlabel('x', fontsize='x-large')
    ax.set_ylabel('y', fontsize='x-large')
    ax.set_zlabel('z', fontsize='x-large')
    plt.show()


def plot2d(t, x, y, r, xlabel, ylabel):
    """
    Plots two input parameters in 2D. t is denoted by color (see colorbar)
    
    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #needed for multicolored-line plotting
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #subplots
    fig, ax = plt.subplots()
    #normalizes the linecolor based on t
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    #sets the linecollection array
    lc.set_array(t)
    lc.set_linewidth(2)
    #plots the line
    line = ax.add_collection(lc)
    #rescales
    ax.autoscale()
    ax.margins(0.1)
    #adds a colorbar for t
    fig.colorbar(line, ax=ax, label='t')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('r=' + str(r))
    plt.show()
    pass

def plot(x, y, title, xlabel, ylabel):
    """
    Generates a simple 2D plot.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor('whitesmoke')
    ax.plot(x,y, color='midnightblue', linewidth=0.75)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)
    plt.show()
    pass

def plot_2(t, x, y, title, xlabel, ylabel, tlabel):
    """
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    tlabel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    color = 'whitesmoke'
    ax.set_facecolor(color)
    ax.grid(True)
    ax.plot(t, x, label=xlabel, color='indigo', linewidth=1)
    ax.plot(t, y, label=ylabel, color='gold', linewidth=1)
    ax.set_xlabel(tlabel)
    ax.set_title(title)
    ax.legend(labelcolor='k', facecolor=color)
    plt.show()
    pass

def plot_3(t, x, y, z, title, xlabel, ylabel, zlabel, tlabel):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    color = "whitesmoke"
    ax.set_facecolor(color)
    ax.plot(t, x, label=xlabel, color='crimson', linewidth=0.75)
    ax.plot(t, y, label=ylabel, color='indigo', linewidth=0.75)
    ax.plot(t, z, label=zlabel, color='gold', linewidth=0.75)
    ax.legend(labelcolor='k', facecolor=color)
    ax.set_title(title)
    ax.set_xlabel(tlabel)
    ax.grid(True)
    plt.show()
    pass
    
def readWave(storage_string):
    """

    Parameters
    ----------
    storage_string : TYPE
        DESCRIPTION.

    Returns
    -------
    signal : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    sample_width : TYPE
        DESCRIPTION.
    sample_rate : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    num_channels : TYPE
        DESCRIPTION.

    """
    #read the wav file at location store_string
    wave_file = wave.open(storage_string, 'rb')
    sample_width = wave_file.getsampwidth()
    sample_rate = wave_file.getframerate()
    #get the number of audio frames
    n = wave_file.getnframes()
    #read all of the grames
    wave_bytes = wave_file.readframes(n)
    #convert the bytes to floats in a np array
    signal = np.frombuffer(wave_bytes, 'int16')
    #return the signal, and number of bytes
    num_channels = wave_file.getnchannels()
    return signal, signal.size, sample_width, sample_rate, n, num_channels

def writeWave(storage_string, signal_bytes, sample_width, sample_rate, numframes):
    wave_file = wave.open(storage_string, 'wb')
    wave_file.setnchannels(2)
    wave_file.setnframes(numframes)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(signal_bytes)
    pass
