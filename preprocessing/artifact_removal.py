import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from feature_extraction.features import hjorth_complexity

def detect_blink_peak(x):
    max_val = np.max(np.absolute(x))
    if max_val >= 0.01:
        return True
    else:
        return False
    
def remove_blinks(x, fs=500/3):
    ica = FastICA(tol=0.1, max_iter=500).fit(x)
    x_ica = ica.transform(x) # applies unmixing matrix to x
    win_length = 200
    for i, col in enumerate(x_ica.T): # iterate columns
        num_blinks = 0
        
        for j in range(0, len(col)-win_length, win_length):
            window = col[j:j+win_length] # window through signal
            is_blink = detect_blink_peak(window) # detect blink in window
        
            # count number of blinks
            if is_blink:
                num_blinks += 1
        
        # compute kurtosis of whole signal
        kur = kurtosis(col)
        
        # compute hjorth complexity
        comp = hjorth_complexity(col)[0][0]
        
        # blink artifact source decision rule
        if num_blinks >= len(x_ica)/(fs*40) and kur > 5 and comp > 3:
            x_ica[:, i] = 0
    
    # apply inverse transformation
    x_clean = ica.inverse_transform(x_ica)
    
    return x_clean

        
    
        
    
    