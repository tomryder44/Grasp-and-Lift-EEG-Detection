
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import welch

from feature_extraction.gal_features import feature_max, feature_min, feature_MAV, hjorth_activity, \
    hjorth_mobility, hjorth_complexity, feature_peak_freq
from feature_extraction.gal_dim_reduction import principal_component_analysis


def compute_feature_row(x):
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
    
def get_features(x, y, window_length, overlap=0.5):
    ''' Pass sliding window over x and extract features. Label of feature is 
    that of the last sample in the window. '''
    feature_rows = [] 
    window_labels = [] # label for each feature
    fs = 500/3
    window_length = int(window_length*fs)
    start = window_length-1
    step = int(window_length-(overlap*window_length))
        
    for i in range(start, len(x), step):
        window_labels.append(y[i,:]) # label given as label of last sample in window
        x_window = x[i-(window_length-1):i+1, :]
        feature_row = compute_feature_row(x_window)
        feature_rows.append(feature_row)
            
    x_fe = np.vstack(feature_rows)
    y_fe = np.vstack(window_labels)
    return x_fe, y_fe
    
def standardise(x, scaler=None):    
    ''' Standardise the train and test sets using mean and std from train set.'''
    if scaler:
        x = scaler.transform(x)
        return x
    else:
        scaler = MinMaxScaler().fit(x)
        x = scaler.transform(x)
        return x, scaler
    
def compute_feature_space(x_train, x_val, x_test, y_train, y_val, y_test,
                       window_length, PCA=False):
    ''' Extract features, normalise and do PCA if specified. '''
    x_train, y_train = get_features(x_train, y_train, window_length) 
    x_val, y_val = get_features(x_val, y_val, window_length) 
    x_test, y_test = get_features(x_test, y_test, window_length)      
    
    # standardise before PCA    
    x_train, scaler = standardise(x_train)
    x_val = standardise(x_val, scaler)
    x_test = standardise(x_test, scaler)
    
    if PCA:
        x_train, pca = principal_component_analysis(x_train)
        x_val = principal_component_analysis(x_val, pca)
        x_test = principal_component_analysis(x_test, pca)

    return x_train, x_val, x_test, y_train, y_val, y_test
    
        
    
    
    
    
    