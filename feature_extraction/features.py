
import numpy as np

def feature_max(x):
    ''' Column-wise maximum value. '''
    col_max = np.max(x, axis=0)
    return col_max.reshape(1,-1)
    
def feature_min(x):
    ''' Column-wise minimum value. '''
    col_min = np.min(x, axis=0)
    return col_min.reshape(1,-1)    

def feature_MAV(x):
    ''' Column-wise mean absolute value. '''
    x_abs = np.absolute(x) 
    x_mav = np.mean(x_abs, axis=0)
    return x_mav.reshape(1,-1) 

def hjorth_activity(x):
    ''' Column-wise computation of Hjorth activity (variance). '''
    return np.var(x, axis=0).reshape(1,-1)
    
def hjorth_mobility(x):
    ''' Column-wise computation of Hjorth mobility'''
    return np.sqrt(np.var(np.gradient(x, axis=0), axis=0)/np.var(x, axis=0)).reshape(1,-1)

def hjorth_complexity(x):
    ''' Column-wise computation of Hjorth complexity'''
    return hjorth_mobility(np.gradient(x, axis=0)) / hjorth_mobility(x).reshape(1,-1)
    
def covmat(x):
    ''' Compute covariance between the channels and take lower triangle elements. '''
    covmatrix = np.cov(x, rowvar=False) # rows contain observations
    idx = np.triu_indices(covmatrix.shape[0])
    covmat_features = covmatrix[idx]
    return covmat_features.reshape(1,-1)
        
def feature_peak_freq(psd, freqs):
    ''' Frequency of the column-wise peak power. '''
    peak_idx = np.argmax(psd, axis=0)
    peak_freq = np.array([freqs[peak_idx[i]] for i in range(len(peak_idx))])
    return peak_freq.reshape(1,-1)
