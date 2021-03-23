
import time
import datetime
import pandas as pd
import numpy as np

from preprocessing.data import load_subject_data
from preprocessing.filter import downsample, decimate, filter_bank
from preprocessing.artifact_removal import remove_blinks

from feature_extraction.extraction import feature_space, normalise, compute_features
from feature_extraction.dim_reduction import principal_component_analysis

from classification.train import optimise_model

from tabulate import tabulate
from sklearn.metrics import roc_auc_score
import json

# set subjects, algorithms and grid in test
subjects = range(9, 12) # (1, 13)
algorithms = [9, 41] # (1, 65)
grid = np.logspace(-1.3, 1.3, 10)

# algorithm results dataframe
t = pd.DataFrame(index=algorithms, columns=['ICA', 'Filter', 'Window length', 
                                            'PCA', 'Penalty', 'Train time', 
                                            'No. feats', 'AUROC'])
t.index.rename('algorithm', inplace=True)
    
for algorithm in algorithms:
    
    # load algorithm parameters
    algorithm_file = 'C:/Users/tomry/Documents/EEG/algorithms/'+ \
                                            'algorithm_'+str(algorithm)+'.txt'
    with open(algorithm_file) as json_file:
        alg_params = json.load(json_file)
    
    ICA = alg_params['ICA']
    filt = alg_params['Filter']
    win_length = alg_params['Window length']
    PCA = alg_params['PCA']
    pen = alg_params['Penalty']
    
    # log algorithm parameters
    t.loc[algorithm, 'ICA'] = ICA
    t.loc[algorithm, 'Filter'] = filt
    t.loc[algorithm, 'Window length'] = win_length
    t.loc[algorithm, 'PCA'] = PCA
    t.loc[algorithm, 'Penalty'] = pen
                
    # record train time, number of features and AUROC for each subject
    train_times = []
    num_feats = []
    aurocs = []

    for subject in subjects:
        print('---')
        print('Algorithm %d, Subject %d' % (algorithm, subject))
        print('---')
        print('Offline part')
      
        subj_time_start = time.time()
        
        # load training data
        x_train = load_subject_data(subject, 'data', 1, 4)
        y_train = load_subject_data(subject, 'events', 1, 4)
        
        # load validation data
        x_val = load_subject_data(subject, 'data', 5, 6)
        y_val = load_subject_data(subject, 'events', 5, 6)
    
        # decimate the signals
        x_train = decimate(x_train)
        x_val = decimate(x_val)
        
        # downsample the events
        y_train = downsample(y_train)
        y_val = downsample(y_val)

        if ICA:
            x_train = remove_blinks(x_train)
            x_val = remove_blinks(x_val)
        
        if filt == 'filter bank':
            x_train = filter_bank(x_train)
            x_val = filter_bank(x_val)
    
        # feature extraction  
        x_train, y_train = feature_space(x_train, y_train, win_length)
        x_val, y_val = feature_space(x_val, y_val, win_length)
            
        # standardise before PCA    
        x_train, scaler = normalise(x_train)
        x_val = normalise(x_val, scaler)
        
        if PCA:
            x_train, pca = principal_component_analysis(x_train)
            x_val = principal_component_analysis(x_val, pca)
            
        # optimise the model with a grid search 
        model, train_time = optimise_model(x_train, x_val, y_train, y_val, pen, grid)
        train_times.append(train_time)
        
        # get number of features
        if pen == 'l1': # number of non-zero coefficients
            # classifier contains 6 LR models - take average of nonzerod coeffs
            num_feat = np.mean(np.count_nonzero(model.coef_, axis=1)) 
        else:
            num_feat = x_train.shape[1]
        num_feats.append(num_feat)
            
        subj_time_end = time.time()
        subj_time = subj_time_end-subj_time_start
        subj_time = str(datetime.timedelta(seconds=subj_time))
        print('Time: ', subj_time)
    
        print('Real-time simulation...')
 
        
        # load test data
        x_test = load_subject_data(subject, 'data', 7, 8)
        y_test = load_subject_data(subject, 'events',  7, 8)
        
        # decimate data
        x_test = decimate(x_test)
        
        # downsample events
        y_test = downsample(y_test)
        
        fs = 500/3
        win_length_samples = int(win_length*fs)

        # record predictions
        predictions = []
        true = []

        for i in range(0, len(x_test)-win_length_samples, win_length_samples):
            
            if i >= 10000: # 10000 samples needed for ICA
                x_win = x_test[i:i+win_length_samples, :]
                true.append(y_test[i+win_length_samples-1, :])
                
                # remove artifacts with ICA
                if ICA:
                    ica_data = x_test[i-10000:i+win_length_samples, :]
                    x_clean = remove_blinks(ica_data)
                    x_win = x_clean[-win_length_samples:, :]
                
                # apply filter bank
                if filt == 'filter bank':
                    x_win = filter_bank(x_win)
                
                # extract features
                x_win = compute_features(x_win)
                
                # normalise 
                x_win = normalise(x_win, scaler)
            
                # reduce dimensionality with PCA
                if PCA:
                    x_win = principal_component_analysis(x_win, pca)
            
                # make predictions on test set    
                y_pred = model.predict_proba(x_win)
                predictions.append(y_pred)
            
        # compute area under receiver operator characteristic curve 
        auroc = roc_auc_score(np.vstack(true), np.vstack(predictions))
        print('Test AUROC: %.3f' % auroc)
        aurocs.append(auroc)
        
    # log metrics
    t.loc[algorithm, 'Train time'] = round(np.mean(train_times),3)
    t.loc[algorithm, 'No. feats'] = round(np.mean(num_feats),3)
    t.loc[algorithm, 'AUROC'] = round(np.mean(aurocs),3)


# build table of results
t.sort_values(by=['AUROC'], ascending=False, inplace=True)
table = tabulate(t, t.columns, showindex=False, tablefmt='github')
print(table)
    