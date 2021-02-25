
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from classification.gal_classification import optimise_model
from preprocessing.gal_preprocessing import get_data
from preprocessing.gal_filtering import downsample, causal_IIR_filter

from feature_extraction.gal_feature_extraction import compute_feature_space

subjects = range(1, 13)
win_length = 0.5 
model = OneVsRestClassifier(LogisticRegression(max_iter=2000))
grid = np.logspace(-1.3, 1.3, 10)

y_pred = []
for subject in subjects:
    print('---')
    print('Subject %d' % subject)
    print('---')
    x_train, x_val, x_test, y_train, y_val = get_data(subject, for_kaggle=True)

    x_train = causal_IIR_filter(x_train, [0.5, 80])
    x_val = causal_IIR_filter(x_val, [0.5, 80])
    x_test = causal_IIR_filter(x_test, [0.5, 80])
    
    x_train = downsample(x_train)
    x_val = downsample(x_val)
    y_train = downsample(y_train)
    y_val = downsample(y_val)
    
    x_train, x_val, x_test, y_train, y_val = compute_feature_space(
                                                    x_train, x_val, x_test, 
                                                    y_train, y_val, win_length)
    
    model, _ = optimise_model(x_train, x_val, y_train, y_val, model, grid)
    
    subject_predictions = []
    for i in range(0,3):
        x_train, x_val, x_test, y_train, y_val = get_data(subject, for_kaggle=True)
        x_test = downsample(x_test, start=i)
        x_train, x_val, x_test, y_train, y_val = compute_feature_space(
                                                    x_train, x_val, x_test, 
                                                    y_train, y_val, win_length)
        y_pred = model.predict_proba(x_test)
        subject_predictions.append(y_pred)
    
    a, b, c = subject_predictions
    subj_y_pred = np.array([row for row_group in zip(a, b, c) for row in row_group])
    y_pred.append(subj_y_pred)

#%%

all_ids = []    
path = r'C:/Users/tomry/Documents/EEG/test/'
series = range(9, 11)
for subject in subjects:
    for serie in series:
        data_filename = path + 'subj%d_series%d_%s.csv' % (subject, serie, 'data')
        data = pd.read_csv(data_filename)
        ids = data['id']
        all_ids.append(ids)        
index = np.concatenate(all_ids)       
  
columns = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']  
    
submission = pd.Dataframe(data=preds,
                          index=index,
                          columns=columns)

#submission.to_csv(submission_file,index_label='id',float_format='%.5f') 