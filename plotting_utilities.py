# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 9:58:3 2021

Contains plotting utilities.

@author: Kade
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot(x: np.array, y: np.array, title: str, xlabel: str, ylabel: str):
    """
    Generates a simple 2D plot. Uses the midnightblue color.

    Parameters
    ----------
    x : np.array
        An array containing x values to plot.
    y : nparray
        An array containing y values to plot.
    title : str
        the title of the graph.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y axis.

    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor('whitesmoke')
    ax.plot(x,y, color='midnightblue', linewidth=0.75)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def plot2d(t, x, y, r, xlabel, ylabel):
    """
    Plots two input parameters against eachother in 2D. 
    t is denoted by color (see colorbar).
    Uses the "plasma" colormap.
    
    Parameters
    ----------
    t : np.array
        An array of time, t.
    x : np.array
        An array of x coordinates.
    y : np.array
        An array of y coordinates.
    r : int
        The r-value for the system.
    xlabel : str
        The label for the x-axist.
    ylabel : str
        The label for the y-axis.

    """
    # needed for multicolored-line plotting
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # subplots
    fig, ax = plt.subplots()
    # normalizes the linecolor based on t
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    # sets the linecollection array
    lc.set_array(t)
    lc.set_linewidth(2)
    # plots the line
    line = ax.add_collection(lc)
    # rescales
    ax.autoscale()
    ax.margins(0.1)
    #adds a colorbar for t
    fig.colorbar(line, ax=ax, label='t')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('r=' + str(r))
    plt.show()
    
    
def plot2d_two_params(t: np.array, x: np.array, y: np.array, title: str,
                      xlabel: str, ylabel: str, tlabel: str):
    """
    Plots two parameters (x and y) against t in 2D space.

    Parameters
    ----------
    t : np.array
        An array of t values against which to plot.
    x : np.array
        An array of x values.
    y : np.array
        An array of y values.
    title : str
        The title of the graph.
    xlabel : str
        The label of the x values.
    ylabel : str
        The label of the y values.
    tlabel : str
        The lable of the t values.

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
    
    
def plot2d_three_params(t: np.array, x: np.array, y: np.array, z: np.array, 
                        title: str, xlabel: str, ylabel: str, zlabel: str, 
                        tlabel: str):
    """
    Plots three parameters (x, y, and z) against t in 2D space.

    Parameters
    ----------
    t : np.array
        An array containing time values.
    x : np.array
        An array containing x values.
    y : np.array
        An array containing y values.
    z : np.array
        An array containing z values.
    title : str
        The title of the graph.
    xlabel : TYPE
        The label for the x values.
    ylabel : TYPE
        The label for the y values.
    zlabel : TYPE
        The label for the z values.
    tlabel : str
        The label for t values.

    """
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
    
    
def plot3d(x: np.array, y: np.array, z: np.array, r: int):
    """
    Plots x, y, z in 3D. Also lists the R values. t is denoted by color,
    but the colormap is cyclical so the color repeats. 
    Prints using the "twilight" colormap.
    
    Parameters
    ----------
    x: np.array
        An array of x coordinates.
    y: np.rray
        An array of y coordinates.
    z: np.array
        An array of z coordinates.
    r: int
        The r-value for this set (to be displayed as the title)
        
    """
    # set up the figure
    fig = plt.figure(figsize = (10,10), dpi=80)
    # print(plt.cm.get_cmap('inferno').N)
    ax = plt.axes(projection='3d')
    # plots in 3D. For loop was necessary for adusting the color of the line
    for i in range(len(x)-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.twilight((i+1)%510))
    # set values and labels
    r_val = 'r= ' + str(r)
    ax.text(-20, 30, 30, r_val, fontsize='x-large')
    ax.set_xlabel('x', fontsize='x-large')
    ax.set_ylabel('y', fontsize='x-large')
    ax.set_zlabel('z', fontsize='x-large')
    plt.show()
