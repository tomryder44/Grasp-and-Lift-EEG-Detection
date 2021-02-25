
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import welch

from feature_extraction.features import feature_max, feature_min, feature_MAV, hjorth_activity, \
    hjorth_mobility, hjorth_complexity, feature_peak_freq

def normalise(x, scaler=None):    
    ''' Normalise the train and test sets using mean and std from train set.'''
    if scaler:
        x = scaler.transform(x)
        return x
    else:
        scaler = MinMaxScaler().fit(x)
        x = scaler.transform(x)
        return x, scaler

def compute_features(x):
    ''' Extract features from columns of x. Feature are concatenated
    along axis 1. '''
    
    # time domain features
    x_max = feature_max(x)
    x_min = feature_min(x)
    x_mav = feature_MAV(x)
    x_HA = hjorth_activity(x)
    x_HM = hjorth_mobility(x)
    x_HC = hjorth_complexity(x)
    #x_cov = covmat(x)
    
    # frequency domain features
    freqs, psd = welch(x, window='hann', axis=0, nperseg=len(x)/2)
    psd_max = feature_max(psd)
    psd_pf = feature_peak_freq(psd, freqs)
    psd_q1_mean = np.mean(psd[0:round(len(psd)/4),:], axis=0).reshape(1,-1)
    psd_q2_mean = np.mean(psd[round(len(psd)/4):round(len(psd)/2)]).reshape(1,-1)
    psd_q3_mean = np.mean(psd[round(len(psd)/2):round(3*len(psd)/4)]).reshape(1,-1)
    psd_q4_mean = np.mean(psd[round(3*len(psd)/4):round(len(psd))]).reshape(1,-1)
    
    # concatenate all extracted features
    feature_row = np.concatenate((x_max, x_min, x_mav, x_HA, x_HM, x_HC,
                        psd_max, psd_pf, psd_q1_mean, psd_q2_mean, psd_q3_mean,
                        psd_q4_mean), axis=1) 
    return feature_row
    
def feature_space(x, y, win_length_s, overlap=0.5):
    ''' Pass sliding window over x and extract features. If y (labels) is specified, 
    label of each window given as label of last sample in that window. '''
    
    fs = 500/3
    win_length_samples = int(win_length_s*fs)
    start = win_length_samples-1
    
    if overlap < 1:
        step = int(win_length_samples-(overlap*win_length_samples))
    else:
        step = overlap
    
    feature_rows = [] 
    window_labels = [] # label for each feature
    
    for i in range(start, len(x), step):
        
        if y is not None: # todo - for test data with no y
            window_labels.append(y[i,:]) # label given as label of last sample in window
        
        # get window of data and extract features
        x_window = x[i-(win_length_samples-1):i+1, :]
        feature_rows.append(compute_features(x_window))
            
    x_fe = np.vstack(feature_rows)
    
    if y is not None:
        y_fe = np.vstack(window_labels)
        return x_fe, y_fe
    
    else:
        return x_fe
        


    