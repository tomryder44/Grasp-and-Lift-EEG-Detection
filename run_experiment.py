
import time
import datetime
import pandas as pd
import numpy as np

from preprocessing.data import get_data
from preprocessing.filter import downsample, decimate, filter_bank
from preprocessing.artifact_removal import independent_component_analysis

from feature_extraction.extraction import feature_space, normalise
from feature_extraction.dim_reduction import principal_component_analysis

from classification.train import optimise_model

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
import json

# set subjects, algorithms and grid in test
subjects = range(1, 2) # (1, 13)
algorithms = range(1, 2) # (1, 65)
grid = np.logspace(-1.3, 1.3, 10)

# algorithm results dataframe
t = pd.DataFrame(index=algorithms, columns=['ICA', 'Filter', 'Window length', 
                                            'PCA', 'Penalty', 'Train time', 
                                            'No. feats', 'AUROC'])
t.index.rename('algorithm', inplace=True)

# preprocessing carried out on all datasets seperately so that ICA and
# subsequent visual inspection can be done at one time

subject_datasets = []
subject_datasets_ica = []
for subject in subjects:
    
    x_train, x_val, x_test, y_train, y_val, y_test = get_data(subject)
    
    x_train = decimate(x_train)
    x_val = decimate(x_val)
    x_test = decimate(x_test)
    
    y_train = downsample(y_train)
    y_val = downsample(y_val)
    y_test = downsample(y_test)
    
    subject_datasets.append((x_train, x_val, x_test, y_train, y_val, y_test))
    
    x_train_ica, ica, col = independent_component_analysis(x_train)
    x_val_ica = independent_component_analysis(x_val, ica, col)
    x_test_ica = independent_component_analysis(x_test, ica, col)
    
    subject_datasets_ica.append((x_train_ica, x_val_ica, x_test_ica, 
                                 y_train, y_val, y_test))

# record overall run time
start_time = time.time()

# run each algorithm and record results
for algorithm in algorithms:
    
    # load algorithm parameters
    algorithm_name = 'C:/Users/tomry/Documents/EEG/algorithms/'+ \
                                            'algorithm_'+str(algorithm)+'.txt'
    with open(algorithm_name) as json_file:
        alg_params = json.load(json_file)
    ICA = alg_params['ICA']
    filt = alg_params['Filter']
    win_length = alg_params['Window length']
    PCA = alg_params['PCA']
    pen = alg_params['Penalty']
                
    # record train time, number of features and AUROC for each subject, 
    # take average for algorithm
    train_times = []
    num_feats = []
    aurocs = []

    # using datasets with ICA for artifact removal or not
    if ICA:
        datasets = subject_datasets_ica
    else:
        datasets = subject_datasets

    for subject, data in enumerate(datasets,1):
        print('---')
        print('Algorithm %d, Subject %d' % (algorithm, subject))
        print('---')
        
        subj_time_start = time.time()
        
        # unpack subject data
        x_train, x_val, x_test, y_train, y_val, y_test = data
        
        # apply filter bank if in algorithm
        if filt == 'filter bank':
            x_train = filter_bank(x_train)
            x_val = filter_bank(x_val)
            x_test = filter_bank(x_test)
    
        # feature extraction  
        x_train, y_train = feature_space(x_train, y_train, win_length)
        x_val, y_val = feature_space(x_val, y_val, win_length)
        x_test, y_test = feature_space(x_test, y_test, win_length)
            
        # standardise before PCA    
        x_train, scaler = normalise(x_train)
        x_val = normalise(x_val, scaler)
        x_test = normalise(x_test, scaler)
        
        if PCA:
            x_train, pca = principal_component_analysis(x_train)
            x_val = principal_component_analysis(x_val, pca)
            x_test = principal_component_analysis(x_test, pca)

        # set regularisation penalty 
        if pen == 'l2':
            model = OneVsRestClassifier(LogisticRegression(max_iter=2000))  
        elif pen == 'l1':
            model = OneVsRestClassifier(LogisticRegression(max_iter=2000,
                                                           penalty='l1',
                                                           solver='saga',
                                                           tol=0.01)) # keeps training time down

        # optimise the model with a grid search 
        model, train_time = optimise_model(x_train, x_val, y_train, y_val, 
                                           model, grid)
        
        # get number of features
        if pen == 'l1': # number of non-zero coefficients
            # classifier contains 6 LR models - take average of nonzerod coeffs
            num_feat = np.mean(np.count_nonzero(model.coef_, axis=1)) 
        else:
            num_feat = x_train.shape[1]
            
        # make predictions on test set    
        y_pred = model.predict_proba(x_test)
        
        # compute area under receiver operator characteristic curve 
        auroc = roc_auc_score(y_test, y_pred)
        print('Test AUROC: %.3f' % auroc)
        
        subj_time_end = time.time()
        subj_time = subj_time_end-subj_time_start
        subj_time = str(datetime.timedelta(seconds=subj_time))
        print('Time: ', subj_time)
    
        # add subject metrics
        train_times.append(train_time)
        num_feats.append(num_feat)
        aurocs.append(auroc)
        
    # log algorithm parameters
    t.loc[algorithm, 'ICA'] = ICA
    t.loc[algorithm, 'Filter'] = filt
    t.loc[algorithm, 'Window length'] = win_length
    t.loc[algorithm, 'PCA'] = PCA
    t.loc[algorithm, 'Penalty'] = pen
            
    # log metrics
    t.loc[algorithm, 'Train time'] = round(np.mean(train_times),3)
    t.loc[algorithm, 'No. feats'] = round(np.mean(num_feats),3)
    t.loc[algorithm, 'AUROC'] = round(np.mean(aurocs),3)

# build table of results
t.sort_values(by=['AUROC'], ascending=False, inplace=True)
table = tabulate(t, t.columns, showindex=False, tablefmt='github')
print(table)
    
# record overall run time
end_time = time.time()
time_seconds = end_time - start_time
overall_time = str(datetime.timedelta(seconds=time_seconds))
print('Overall run time: ', overall_time)




