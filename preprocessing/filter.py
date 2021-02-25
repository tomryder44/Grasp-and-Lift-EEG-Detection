
import numpy as np
from scipy.signal import lfilter, butter

def downsample(x, n=3, start=0):
    ''' Downsamples the signals in x by taking every nth sample. '''
    return x[start::n, :]

def causal_filt(x, cutoffs, fs=500, btype='bandpass'):
    ''' Applies a causal IIR filter to x. '''
    b, a = butter(4, cutoffs, fs=fs, btype=btype)
    x_filtered = lfilter(b, a, x, axis=0)
    return x_filtered
    
def filter_bank(x):
    ''' Computes the FIR filter coefficients and applies causal filtering 
    with lfilter. Concatenates the filtered signals in axis 1. '''
    # frequency ranges
    delta = (0.5, 4)
    theta = (4, 8)
    alpha = (8, 13)
    beta = (13, 30)
    gamma = (30, 80) 
    bank = (delta, theta, alpha, beta, gamma)
    x_filtered = []
    for cutoffs in bank:
        cutoffs = [cutoffs[0], cutoffs[1]]
        b, a = butter(4, cutoffs, fs=500/3, btype='bandpass')
        x_filt = lfilter(b, a, x, axis=0)
        x_filtered.append(x_filt)
    x_filtered = np.concatenate(x_filtered, axis=1)
    return x_filtered

def decimate(x):
    ''' Decimate the signal by filtering and downsampling. '''
    x = causal_filt(x, [0.5, 80])
    x = downsample(x)
    return x