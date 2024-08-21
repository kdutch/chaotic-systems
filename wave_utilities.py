#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 20:02:30 2024
Contains utilities that read and write waves from WAV files.

@author: Kade
"""
import wave
import numpy as np

from pathlib import Path
from typing import Tuple, Union


def read_wave(wave_file_path: Union[Path, str]) -> \
    Tuple[np.array, int, bytes, float, int]:
    """
    Reads a wave from a provided file_path and then decomposes it into 
    a numpy array and returns the parameters of the wave.
    
    Parameters
    ----------
    wave_file_path : Union[Path, str]
        The file_path for the location of the WAV file to read.

    Returns
    -------
    signal : np.array
        The numpy array representation oft the read wave.
    signal.size: int
        The size of the array.
    sample_width : bytes
        The width of the signal in bytes.
    sample_rate : float
        The sample freq of the wave.
    n : int
        The number of frames in the wave.
    num_channels : int
        The number of channels in the wave file.

    """
    # read the wav file at location store_string
    with wave.open(str(wave_file_path), 'rb') as wave_file:
        sample_width = wave_file.getsampwidth()
        sample_rate = wave_file.getframerate()
        # get the number of audio frames
        n = wave_file.getnframes()
        # read all of the frames
        wave_bytes = wave_file.readframes(n)
        # convert the bytes to floats in a np array
        signal = np.frombuffer(wave_bytes, 'int16')
        # return the signal, and number of bytes
        num_channels = wave_file.getnchannels()
    return signal, signal.size, sample_width, sample_rate, n, num_channels


def write_wave(wave_file_path: Union[Path, str], signal_bytes: np.array,
               sample_width: int, sample_rate: float, numframes: int):
    """
    Writes the wave to a WAV to the provided wave_file_path.

    Parameters
    ----------
    wave_file_path : Union[Path, str]
        The destination file_path of the wave.
    signal_bytes : np.array
        A numpy array containing the bytes to write to the WAV file.
    sample_width : bytes
        The sample width of the wave in bytes.
    sample_rate : float
        The sameple frequency of the wave.
    numframes : the number of frames.
        The number of frames in the sample.

    """
    with wave.open(wave_file_path, 'wb') as wave_file:
        wave_file.setnchannels(2)
        wave_file.setnframes(numframes)
        wave_file.setsampwidth(sample_width)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(signal_bytes)
