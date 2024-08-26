# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:03:07 2021

Contaings all of the methods and functions associated with synchronizing two 
systems with (a) starting output and (b) a perterbed distribution.

The perterbation can be anything, including a wave-like signal you would like 
to "transmit". Chaotic system synchronization therefore has potential in 
cryptography for signal masking.

The below code is an exploration of that.

@author: Kade
"""
import argparse
import logging

import numpy as np

from dataclasses import dataclass, field
from pathlib import Path

from integrators import SynchronizedRK45
from functions import lorenz, reciever
from plotting_utilities import plot, plot2d_three_params, plot2d_two_param_sets
from wave_utilities import read_wave, write_wave


@dataclass
class WaveProperties:
    """
    A dataclass containing any properties associated with the imported wav.
    
    signal_raw: np.array
        An array containing the bytes in the wav file.
    signal_size: tuple, int
        The size of the array
    sample_width: bytes
        The width of each sample
    sample_rate: float
        The frequency of the samples.
    num_frames: int
        The number of frames in the wav file.
    num_channels: int
        The number of channels int the wav file.
    sample_time: float
        The amount of time per sample, calculated by 1/sample_rate.
    time: float
        The total time of the sample, calculated by the sample_time multiplied 
        by the number of frames.
    
    """
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
    A dataclass containing all of the conditions for system. This includes:
        
    y0: np.array
        The initial condition for the Lorenz system. Can be editted.
    r_values: list
        The r_values with which to test how the system can be transmitted.
    wave_file_path: Path
        The file_path for the "wav" file containing the data to "transmit".
    percent: float
        The normalization percentage for the wave.
    max_steps: int
        The max_steps for the integration.
    wave_properties: WaveProperties
        The properties of the wav to be transmitted.
    t0: float
        starting point of the integration (to give time for the system to 
                                           synchronise).
    t1: float
        End point for the integration (the length of the imported wav)
    tol: float = np.inf
        The tolerance for the integration. Cannot be changed because step-size
        is locked.
    
    """
    y0: np.array = field(default_factory=list)
    r_values: list = field(default_factory=list)
    wave_file_path: Path = field(default="wav_files/song_of_storms.wav")
    # synchronixation normalization conditions
    percent: float = field(default=0.01)
    max_steps: int = field(default_factory=int)
    offset: float = field(default=2.5)
    tol: float = field(default=10**(-6))
    wave_properties: WaveProperties = field(default=None, 
                                            init=False)
    t0: float = field(default=0, init=False)
    t1: float = field(default_factory=int, init=False)
    norm: float = field(default_factory=float, init=False)
    sig: float = field(default=10, init=False)
    b: float = field(default=8/3, init=False)
    h: float = field(default_factory=float, init=False)
    check_sync: bool = field(default=False)
    transmit_signal: bool = field(default=False)
    
    def __post_init__(self):
        """
        Ensures all optional arguments are of the correct type, and processes
        them accordingly. Also calculates and sets indirect system properties
        (such as the normalization or integration step-size).
        
        Raises
        ------
        ValueError
            If any of the input arguments are of an unacceptable type.

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
        self.h = self.wave_properties.sample_time
        self.tol = np.inf
        self.t0 = -self.offset
        if not self.max_steps:
            self.max_steps = int(self.t1/self.h + (self.offset/self.h)) + 1
            
    
def parse_input_arguments() -> SystemConditions:
    """
    Parses input arguments and then returns a SystemConditions object 
    containing all of the conditions of the system 
    
    Returns
    -------
    system_parameters: SystemConditions
        An instance of the SystemConditions dataclass containing all 
        applicable system parameter information.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-values", nargs='*', 
                        action='store', required=False, type=float)
    parser.add_argument("--wave-file-path", action='store', required=False,
                        type=Path, default="wav_files/song_of_storms.wav")
    parser.add_argument("--y0", action='store', nargs=3, required=False,
                        type=float)
    parser.add_argument("--percent-wave-amp", action='store', 
                        required=False, type=float, default=0.01)
    parser.add_argument("--offset", action='store', required=False,
                        type=float)
    parser.add_argument("--check-for-synchronization-conditions", 
                        required=False, action='store_true')
    parser.add_argument("--transmit-signal", required=False,
                        action='store_true')
    parsed_arguments = parser.parse_args()
    system_parameters = process_input_arguments(parsed_arguments)
    return system_parameters
    
    
def process_input_arguments(parsed_args: argparse.Namespace) \
    -> SystemConditions:
    """
    Parses applicable arguments from the parsed_args Namespace 
    and returns a SystemConditions object with the given parameters.
    
    Parameters
    ----------
    parsed_args: argparse.Namespace
        The namespace for the defined input arguments.
        
    Returns
    -------
    SystemConditions
        A SystemConditions object containing the parsed input parameters.
    
    """
    arguments = {}
    if arg := parsed_args.r_values:
        arguments["r_values"] = arg
    if arg:= parsed_args.wave_file_path:
        arguments["wave_file_path"] = arg
    if arg := parsed_args.offset:
        arguments["offset"] = arg
    if arg := parsed_args.y0:
        arguments["y0"] = arg
    if arg := parsed_args.percent_wave_amp:
        arguments["percent"] = arg
    if arg := parsed_args.check_for_synchronization_conditions:
        arguments["check_sync"] = arg
    if arg := parsed_args.transmit_signal:
        arguments["transmit_signal"] = arg
    return SystemConditions(**arguments)
    

def check_for_synchronization_conditions(system_parameters: SystemConditions):
    """
    Checks for the values for which r is synchronized depending on the input 
    parameters by performing system synchronization with provided input 
    arguments.
    
    Parameters
    ----------
    integrator: SynchronizedRK45
        The integrator to use for the integrations.
    system_parameters: SystemConditions
        The dataclass containing system information.
        
    """
    integrator = SynchronizedRK45(
        tol=max(system_parameters.wave_properties.signal_raw)*
        system_parameters.norm, max_steps=system_parameters.max_steps)
    for r in system_parameters.r_values:
        res, t, h = integrator.integrate(lorenz, 
                                         system_parameters.t0, 
                                         system_parameters.t1,
                                         system_parameters.y0, 
                                         system_parameters.h*10**2,
                                         system_parameters.h,
                                         system_parameters.sig,
                                         r, 
                                         system_parameters.b)
        x, y, z = res[:,0], res[:,1], res[:,2]
        synced, t2, h = integrator.synchronize(reciever, 
                                     system_parameters.t0, 
                                     system_parameters.t1, 
                                     100*np.random.rand(3),
                                     system_parameters.h*10**2, 
                                     system_parameters.h,
                                     x,
                                     system_parameters.sig,
                                     r, 
                                     system_parameters.b)
        #len = min(len(u), len(v), len(w), len(x), len(y), len(z))
        u, v, w = synced[:,0], synced[:, 1], synced[:, 2]
        title =f"r={str(r)} | tol= {str(integrator.tol)}"
        plot2d_two_param_sets(t, t2, x, u, "x vs. u", "x", "u", "time")
        plot2d_two_param_sets(t, t2, y, v, "y vs. v", "y", "v", "time")
        plot2d_two_param_sets(t, t2, z, w, "z vs. w", "z", "w", "time")
        plot2d_three_params(t, np.abs(x-u), np.abs(y-v), np.abs(z-w), title,
                           '|x-u|', '|y-v|', '|z-w|', 't')
        plot(t, np.abs(x-u), title, 't', '|x-u|')
        plot(t, np.abs(y-v), title, 't',  '|y-v|')
        plot(t, np.abs(z-w), title, 't', '|z-w|')
        
    
def print_wave(norm: float, wave_properties: WaveProperties, 
               signal_raw: np.array, title: str):
    """
    Prints the provided waveform.

    Parameters
    ----------
    norm : float
        The normalization value of the wave.
    wave_properties : WaveProperties
        Dataclass containing wave properties, such as the signal size,
        the number of channels, etc.
    title: str
        The title for the plot.

    """
    signal = signal_raw
    i = np.linspace(0, wave_properties.signal_size,
                    wave_properties.num_frames * wave_properties.num_channels)
    plot(i, signal, title, 'frame', 'byte (int)')


def integrate_and_plot_original_distribution(integrator: SynchronizedRK45,
                                             system_parameters:
                                                 SystemConditions,
                                             r: int):
    """
    Integrates and plots the distribution before signal perterbation.
    
    Parameters
    ----------
    integrator: SynchronizedRK45
        The intgerator to use when integrating the system
    system_parameters: SystemConditions
        A dataclass containing all system conditions for this run.
    r: int
        The r-value to use for the integration.
        
    """
    #integrate first system
    res, t, h = integrator.integrate(
        lorenz, 
        system_parameters.t0, 
        system_parameters.t1, 
        system_parameters.y0, 
        system_parameters.h, 
        system_parameters.h, 
        system_parameters.sig,
        r, 
        system_parameters.b)
    # plot the original distribution
    x = res[:,0]
    plot(t, x, f"original distribution | r={str(r)}", 't', 'x')
    return x, t


def perterb_and_plot_distribution(x: np.array, t: np.array, 
                                  system_parameters: SystemConditions,
                                  signal_raw: np.array, r: int):
    """
    Perterbs the original distribution with the waveform and pltos the 
    resulting distribution.
    
    Parameters
    ----------
    x: np.array
        The axis to perterb.
    system_parameters: SystemConditions
        The conditions for the system.
    signal_raw: np.array
        The signal we are adding to the distribution.
    r: int
        The r-value for this run.
        
    Returns
    -------
    index: tuple
        A tuple containing all values where t > 0 (where the signal was added).
    X_t: np.array
        The perterbed signal.
        
    """
    #perturb x(t)
    X_t = np.zeros(t.size)
    
    # calculates the indices to match the values for whcih t > 0
    index = np.where( t >= 0)
    if len(index[0]) > len(signal_raw):
        index[0] = index[0][:len(signal_raw)]
    elif len(signal_raw) > len(index[0]):
        signal_raw = signal_raw[:len(index[0])]
    
    X_t[index] = system_parameters.norm * signal_raw
    X_t = X_t + x
    plot(t, X_t, 
         f"perterbed distribution | r={str(r)} | "
         f"norm={system_parameters.percent}%", 
        't', 'X(t)')
    return index, X_t
    

def synchronize_systems(integrator: SynchronizedRK45,
                        system_parameters: SystemConditions,
                        perterbed_signal: np.array, r: int,
                        index: tuple):
    """
    Synchonizes a blank system with a random starting value with the perterbed
    signal.
    
    Parameters
    ----------
    integrator: SycnhronizedRK45
        The integrator to use.
    system_parameters: SystemConditions
        The conditions and values for this run.
    perterbed_signal: np.array
        The "received" perterbed signal.
    r: int
        The r-value for this run.
    index: tuple
        The indices where t> 0

    """
    #synchronize the system
    y, t, h = integrator.synchronize(f=reciever, 
                                     t0=system_parameters.t0, 
                                     t1=system_parameters.t1, 
                                     y0=100*np.random.rand(3), 
                                     hmax=system_parameters.h, 
                                     hmin=system_parameters.h, 
                                     X=perterbed_signal,
                                     sig=system_parameters.sig, 
                                     r=r, 
                                     b=system_parameters.b)
    u, _, _, = y[:, 0], y[:, 1], y[:,2]
    signal_recieved = perterbed_signal - u
    signal_recieved = signal_recieved[index]
    signal_sent = signal_recieved/system_parameters.norm
    print_wave(norm=system_parameters.norm, 
               wave_properties=system_parameters.wave_properties,
               signal_raw=signal_sent, 
               title=f"sent signal | r={str(r)}  | norm="
                     f"{str(system_parameters.percent)} %")
    return signal_sent
    

def export_signal(signal_sent: np.array, system_parameters: SystemConditions, 
                  r: int):
    """
    Exports the signal to an external folder so that you can comapare 
    the audio of the signals to check the quality.
    
    Parameters
    ----------
    signal_sent: np.array
        The signal recieved by subtracting the synchronized values from the 
        perterbed values
    system_parameters: SystemConditions
        The system conditons for this run.
        
    """
    signal_decoded = signal_sent.astype('int16').tobytes()
    num_frames = signal_sent.size
    
    file_string = f"wav_files/sent_wavs/"\
                  f"{str(system_parameters.wave_file_path.stem)}"\
                  f"_r={str(r)[0:4]}_h={str(system_parameters.h)}"\
                  f"_norm={str(system_parameters.norm)}.wav"
    write_wave(file_string, signal_decoded, 
               system_parameters.wave_properties.sample_width, 
               system_parameters.wave_properties.sample_rate, 
               num_frames)


def main():
    system_parameters = parse_input_arguments()
    integrator = SynchronizedRK45(tol=system_parameters.tol, 
                                  max_steps=system_parameters.max_steps)
    # check for system_conditions system
    if system_parameters.check_sync:
        check_for_synchronization_conditions(system_parameters)
        
    if system_parameters.transmit_signal:
        # transmit wave
        print_wave(norm=system_parameters.norm, 
                   wave_properties=system_parameters.wave_properties,
                   signal_raw=system_parameters.wave_properties.signal_raw,
                   title="Original WAV")
    
        for r in system_parameters.r_values:
       
            #integrate first system
            x, t, = integrate_and_plot_original_distribution(
                integrator=integrator, system_parameters=system_parameters, 
                r=r)
            
            #perturb x(t)
            index, X_t = perterb_and_plot_distribution(
                x=x, t=t, system_parameters=system_parameters, 
                signal_raw=system_parameters.wave_properties.signal_raw, r=r)
            
            #synchronize the system
            signal_sent = synchronize_systems(integrator=integrator, r=r, 
                                              system_parameters=
                                              system_parameters, 
                                              perterbed_signal=X_t,
                                              index=index)
            
            # write the new wave
            export_signal(signal_sent, system_parameters, r)

main()
