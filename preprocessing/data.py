
import pandas as pd
import numpy as np

def load_subject_data(subject, filetype, start_series, end_series):
    ''' Loads the data or events specified as filetype for a subject. The series
    are specified to build a training and testing sets. '''
    path = r'C:/Users/tomry/Documents/EEG/train/'
    series = range(start_series, end_series+1)
    subject_data = []
    for serie in series:
        data_filename = path + 'subj%d_series%d_%s.csv' % (subject, serie, filetype)
        data = pd.read_csv(data_filename)
        subject_data.append(data)
    subj_data = pd.concat(subject_data)
    subj_data.drop('id', inplace=True, axis=1)
    return subj_data.to_numpy()

def load_test_data(subject):
    path = r'C:/Users/tomry/Documents/EEG/test/'
    series = range(9, 11)
    subject_data = []
    for serie in series:
        data_filename = path + 'subj%d_series%d_data.csv' % (subject, serie)
        data = pd.read_csv(data_filename)
        subject_data.append(data)
    subj_data = pd.concat(subject_data)
    subj_data.drop('id', inplace=True, axis=1)
    return subj_data.to_numpy()
           
def get_data(subject, for_kaggle=False):
    ''' Gets the data and events for training and testing sets for a subject 
    using load_subject_data. '''
    if for_kaggle:
        x_train = load_subject_data(subject, 'data', 1, 6)
        x_val = load_subject_data(subject, 'data', 7, 8)
        x_test = load_test_data(subject)
        y_train = load_subject_data(subject, 'events', 1, 6)
        y_val = load_subject_data(subject, 'events', 7, 8)
        return x_train, x_val, x_test, y_train, y_val
    else:
        x_train = load_subject_data(subject, 'data', 1, 4)
        x_val = load_subject_data(subject, 'data', 5, 6)
        x_test = load_subject_data(subject, 'data', 7, 8)
        y_train = load_subject_data(subject, 'events', 1, 4)
        y_val = load_subject_data(subject, 'events', 5, 6)
        y_test = load_subject_data(subject, 'events',  7, 8)
        return x_train, x_val, x_test, y_train, y_val, y_test
   