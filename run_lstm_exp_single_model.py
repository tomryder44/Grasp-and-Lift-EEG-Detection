import time
import datetime
import pandas as pd
import numpy as np

from preprocessing.data import load_all_data, load_subject_data, time_series_gen
from preprocessing.filter import downsample, decimate, filter_bank

from feature_extraction.extraction import normalise

from classification.lstm import compile_LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tabulate import tabulate
from matplotlib import pyplot as plt
    
downsample_factor = 8
input_len_s = 0.5
fs = 500/downsample_factor
time_steps = int(input_len_s * fs)
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=10, 
                   min_delta=0.001, restore_best_weights=True)

time_start = time.time()
        
# load training data
x_train = load_all_data('data', 1, 4)
y_train = load_all_data('events', 1, 4)

# load validation data
x_val = load_all_data('data', 5, 6)
y_val = load_all_data('events', 5, 6)
    
# decimate the signals
x_train = decimate(x_train, n=downsample_factor)
x_val = decimate(x_val, n=downsample_factor)

# downsample the events
y_train = downsample(y_train, n=downsample_factor)
y_val = downsample(y_val, n=downsample_factor)

# normalise 
x_train, scaler = normalise(x_train)
x_val = normalise(x_val, scaler)

# get time series batches
train_gen = time_series_gen(x_train, y_train, time_steps)
val_gen = time_series_gen(x_val, y_val, time_steps)

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

time_end = time.time()
time = time_end-time_start
time = str(datetime.timedelta(seconds=time))
time = time.split('.')[0]
print('Time: ', time)

# Testing
subjects = range(1, 13)

t = pd.DataFrame(index=subjects, columns=['AUROC'])
t.index.rename('Subject', inplace=True)

# testing
for subject in subjects:
    print('Testing subject %d' % subject)
    # load testing data
    x_test = load_subject_data(subject, 'data', 7, 8)
    y_test = load_subject_data(subject, 'events',  7, 8)

    x_test = decimate(x_test, n=downsample_factor)
    y_test = downsample(y_test, n=downsample_factor)
    
    x_test = normalise(x_test, scaler)
        
    test_gen = time_series_gen(x_test, y_test, time_steps)
    
    auc = lstm.evaluate(test_gen)[1]

    print('Test AUROC: %.3f' % auc)

    t.loc[subject, 'AUROC'] = round(auc, 3)

# build table of results
cols = ['Subject', 'AUROC']
table = tabulate(t, cols, tablefmt='github')
print(table)