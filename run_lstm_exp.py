import time
import datetime
import pandas as pd
import numpy as np

from preprocessing.data import load_subject_data, time_series_gen
from preprocessing.filter import downsample, decimate

from feature_extraction.extraction import normalise

from classification.lstm import compile_LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tabulate import tabulate
from matplotlib import pyplot as plt

# set subjects, algorithms and grid in test
subjects = range(1, 13)
    
t = pd.DataFrame(index=subjects, columns=['Train time', 'AUROC'])
t.index.rename('Subject', inplace=True)

downsample_factor = 8
input_len_s = 0.5 
fs = 500/downsample_factor
time_steps = int(input_len_s * fs)
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=20, 
                   min_delta=0.001, restore_best_weights=True)

for subject in subjects:
    print('---')
    print('Training LSTM for subject %d' % (subject))
    print('---')
   
    subj_time_start = time.time()
            
    # load training data
    x_train = load_subject_data(subject, 'data', 1, 4)
    y_train = load_subject_data(subject, 'events', 1, 4)
        
    # load validation data
    x_val = load_subject_data(subject, 'data', 5, 6)
    y_val = load_subject_data(subject, 'events', 5, 6)
    
    # load testing data
    x_test = load_subject_data(subject, 'data', 7, 8)
    y_test = load_subject_data(subject, 'events',  7, 8)
    
    # decimate the signals
    x_train = decimate(x_train, n=downsample_factor)
    x_val = decimate(x_val, n=downsample_factor)
    x_test = decimate(x_test, n=downsample_factor)
    
    # downsample the events
    y_train = downsample(y_train, n=downsample_factor)
    y_val = downsample(y_val, n=downsample_factor)
    y_test = downsample(y_test, n=downsample_factor)

    # normalise 
    x_train, scaler = normalise(x_train)
    x_val = normalise(x_val, scaler)
    x_test = normalise(x_test, scaler)
        
    # train LSTM
    train_gen = time_series_gen(x_train, y_train, time_steps)
    val_gen = time_series_gen(x_val, y_val, time_steps)
    test_gen = time_series_gen(x_test, y_test, time_steps)
    
    lstm = compile_LSTM(time_steps)
        
    history = lstm.fit(x=train_gen, epochs=1000, 
                       validation_data=val_gen, 
                       verbose=1, 
                       shuffle=False,
                       callbacks=[es])
    
    plt.plot(history.history['auc'], label='train')
    plt.plot(history.history['val_auc'], label='val')
    plt.legend()
    plt.show()
                
    subj_time_end = time.time()
    subj_time = subj_time_end-subj_time_start
    subj_time = str(datetime.timedelta(seconds=subj_time))
    subj_time = subj_time.split('.')[0]
    print('Time: ', subj_time)

    print('Testing...')
    auc = lstm.evaluate(test_gen)[1]
    
    print('Test AUROC: %.3f' % auc)

    # log metrics
    t.loc[subject, 'Train time'] = subj_time
    t.loc[subject, 'AUROC'] = round(auc, 3)


# build table of results
cols = ['Subject', 'Train time', 'AUROC']
table = tabulate(t, cols, tablefmt='github')
print(table)