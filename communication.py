# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:03:07 2021

@author: Kade
"""
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from integrators import SynchronizedRK45
from functions import lorenz, reciever
from plotting_utilities import plot, plot2d, plot2d_two_params, \
    plot2d_three_params, plot3d
from wave_utilities import read_wave, write_wave


@dataclass
class WaveProperties:
    signal_raw: np.array
    signal_size: int 
    sample_width: int
    sample_rate: float
    num_frames: int
    num_channels: int
    sample_time: float = field(default_factory=float, init=False)
    time: float = field(default_factory=float, init=False)
    
    def __post_init__(self):
        """
        Sets the sample_time and total time of the wave depening on the 
        wave file properties.
        
        """
        self.sample_time = 1/self.sample_rate
        self.time = self.sample_time*self.num_frames
    

@dataclass
class SystemConditions:
    """
    
    """
    ##tol = 
    y0: np.array = field(default_factory=list)
    r_values: list = field(default_factory=list)
    wave_file_path: Path = field(default="wav_files/song_of_storms.wav")
    # synchronixation normalization conditions
    percent: float = field(default=0.1)
    max_steps: int = field(default_factory=int)
    offset: float = field(default=2.5)
    h_min: float = field(default=2.5*10**(-5))
    tol: float = field(default=10**(-6))
    wave_properties: WaveProperties = field(default=None, 
                                            init=False)
    t0: float = field(default=0, init=False)
    t1: float = field(default_factory=int, init=False)
    norm: float = field(default_factory=float, init=False)
    sig: float = field(default=10, init=False)
    b: float = field(default=8/3, init=False)
    h_max: float = field(default_factory=float, init=False)
    
    

    def __post_init__(self):
        """
        

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not self.y0:
            self.y0 = [10, 10, 10]
        if type(self.y0) in (np.array, list) and len(self.y0) == 3:
            for item in self.y0:
                assert isinstance(item, (int, float, np.float64)), \
                    logging.error("All values in y0 must be an int or float." 
                                  "Intstead item [%s] is of type [%s].",
                                  item, type(item))
            self.y0 = np.array(self.y0) 
        else:
            logging.error("y0 was of an unrecognized type [%s] or an invalid" 
                          "length. y0 must be a list or a numpy array of "
                          "floats, and of length 3.", type(self.y0))
            raise ValueError
        if not self.r_values:
            self.r_values = [28]
        if not isinstance(self.tol, float):
            logging.error("tol is of an incorrect type [%s]. tol"
                          " must be a float.", type(self.tol))
            raise ValueError
        if isinstance(self.wave_file_path, Path):
            assert self.wave_file_path.exists(), \
                logging.error("The provided file path [%s] is not a "
                              "valid file-path.")
            self.wave_properties = WaveProperties(
                *read_wave(self.wave_file_path))
        else:
            logging.error("wave_file_path must be a Path object. Instead is "
                          "a [%s] object.", type(self.wave_file_path))
            raise ValueError
        if isinstance(self.percent, float):
            self.norm = self.percent / 100
        else:
            logging.error("percent must be a float. Percent is instead of "
                          "type [%s].", type(self.percent))
            assert ValueError
        # t1 is the total time of the wave multiplied by the number of channels
        self.t1 = self.wave_properties.time * self.wave_properties.num_channels
        self.h_max = self.wave_properties.sample_time
        self.tol = max(self.wave_properties.signal_raw) * self.norm
        self.t0 = -self.offset
        if not self.max_steps:
            self.max_steps = int((self.wave_properties.num_channels * 
                                  self.wave_properties.num_frames) + \
                                 (self.offset/self.h_max))
    
def parse_input_arguments() -> SystemConditions:
    """
    Parses input arguments and then returns a SystemConditions object 
    containing all of the conditions of the system 
    
    Returns
    -------
    system_parameters: SystemConditions

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-values", nargs='*', 
                        action='store', required=False, type=float)
    parser.add_argument("--wave-file-path", action='store', required=False,
                        type=Path, default="wav_files/song_of_storms.wav")
    parser.add_argument("--y0", action='store', nargs=3, required=False,
                        type=float)
    parser.add_argument("--percent-wave-amp", action='store', required=False, type=float,
                       default=0.1)
    parser.add_argument("--hmin", action='store', required=False)
    parser.add_argument("--offset", action='store', required=False)
    parsed_arguments = parser.parse_args()
    system_parameters = process_input_arguments(parsed_arguments)
    return system_parameters
    
    
def process_input_arguments(parsed_args: argparse.Namespace) \
    -> SystemConditions:
    """
    """
    arguments = {}
    if parsed_args.r_values:
        arguments["r_values"] = parsed_args.r_values
    if parsed_args.hmin:
        arguments["hmin"] = parsed_args.hmin
    if parsed_args.wave_file_path:
        arguments["wave_file_path"] = parsed_args.wave_file_path
    if parsed_args.offset:
        arguments["offset"] = parsed_args.offset
    if parsed_args.y0:
        arguments["y0"] = parsed_args.offset
    return SystemConditions(**arguments)
    

def check_for_synchronization_conditions(integrator: SynchronizedRK45,
                                         system_parameters: SystemConditions):
    """
    Checks for the values for which r is synchronized.
    
    """
    #time*k
    # h = 2.5*10**(-5)
    #synchronize the system
    for r in system_parameters.r_values:
        res, t, h = integrator.integrate(lorenz, 
                                         system_parameters.t0, 
                                         system_parameters.t1,
                                         system_parameters.y0, 
                                         system_parameters.hmax,
                                         system_parameters.hmin,
                                         system_parameters.sig,
                                         r, 
                                         system_parameters.b)
        x = res[:,0]
        y = res[:,1]
        z = res[:,2]
        u, v, w, t2, h = integrator.synchronize(reciever, 
                                     system_parameters.t0, 
                                     system_parameters.t1, 
                                     system_parameters.y0 - 
                                     np.array[10, 10, 10],
                                     system_parameters.hmax, 
                                     system_parameters.hmin, 
                                     system_parameters.sig,
                                     r, 
                                     system_parameters.b,
                                     x)
        title ='r=' + str(r) + ' | tol=' + str(integrator.tol)
        plot2d_three_params(t, np.abs(x-u), np.abs(y-v), np.abs(z-w), title,
                            '|x-u|', '|y-v|', '|z-w|', 't')
        plot2d_three_params(t[200000::], np.abs(x-u)[200000::], 
                            np.abs(y-v)[200000::], np.abs(z-w)[200000::], 
                            title, '|x-u|', '|y-v|', '|z-w|', 't')
        plot(t, np.abs(x-u), title, 't', '|x-u|')
        plot(t, np.abs(y-v), title, 't',  '|y-v|')
        plot(t, np.abs(z-w), title, 't', '|z-w|')
        
    
def print_original_wave(norm: float, wave_properties: WaveProperties,
                        title: str):
    """
    

    Parameters
    ----------
    norm : float
        DESCRIPTION.
    wave_properties : WaveProperties
        DESCRIPTION.

    """
    signal = norm*wave_properties.signal_raw
    i = np.linspace(0, wave_properties.signal_size,
                    wave_properties.num_frames * wave_properties.num_channels)
    title = 'original signal'
    plot(i, wave_properties.signal_raw, title, 'frame', 'byte (int)')



def main():
    system_parameters = parse_input_arguments()
    integrator = SynchronizedRK45(tol=system_parameters.tol, 
                                  max_steps=system_parameters.max_steps)
    print_original_wave(norm=system_parameters.norm, 
                        wave_properties=system_parameters.wave_properties,
                        title="Original WAV")
   
    for r in system_parameters.r_values:
   
    #integrate first system
        res, t, h = integrator.integrate(
            lorenz, 
            system_parameters.t0, 
            system_parameters.t1, 
            system_parameters.y0, 
            system_parameters.h_max, 
            system_parameters.h_min, 
            system_parameters.sig,
            r, 
            system_parameters.b)
        x = res[:,0]
        title = 'original distribution | r=' + str(r)
        plot(t, x, title, 't', 'x')
        
        #perturb x(t)
        X_t = np.zeros(t.size)
        index = np.where( t >= 0)
        X_t[index] = system_parameters.norm * \
            system_parameters.wave_properties.signal_raw
        X_t = X_t + x
        title = 'perterbed distribution | r=' + str(r) + '| norm=' + \
            str(system_parameters.percent) + '%'
        plot(t, X_t, title, 't', 'X(t)')
    #synchronize the system
    u, v, w, t, h = integrator.synchronize(f=reciever, 
                                           t0=system_parameters.t0, 
                                           t1=system_parameters.t1, 
                                           y0=system_parameters.y0, 
                                           hmax=system_parameters.h_max, 
                                           hmin=system_parameters.h_min, 
                                           X=X_t,
                                           sig=system_parameters.sig, 
                                           r=r, 
                                           b=system_parameters.b)
    
    #decode the signal
    signal_recieved = X_t - u
    signal_recieved = signal_recieved[index]
    signal_sent = signal_recieved/system_parameters.norm
    title = 'sent signal | r=' + str(r) + ' | norm=' + \
        str(system_parameters.percent) + '%'
    i = np.linspace(0, system_parameters.wave_properties.signal_size,
                    len(signal_sent))
    plot(i, signal_sent, title, 'frame', 'byte (int)')
    signal_decoded = signal_sent.astype('int16').tobytes()
    num_frames = signal_sent.size
    #file_string = 'sent/song_of_storms_01_r=' + str(r)[0:4] + '_h=' + str(hmax) + '_norm=' + str(norm) + '.wav'
    #writeWave(file_string, signal_decoded, sample_width, sample_rate, num_frames)
    #'''
    pass

main()