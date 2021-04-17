
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def load_subject_data(subject, filetype, start_series, end_series):
    ''' Loads the data or events specified as filetype for a subject. The series
    are specified to build a training and testing sets. '''
    keep_channels =['Fp1', 'Fp2', 
                'F7', 'F3', 'Fz', 'F4', 'F8', 
                'FC5', 'FC6', 
                'T7', 'C3', 'C4', 'T8', 
                'CP5', 'CP6',
                'P7', 'P3', 'P4', 'P8', 
                'O1', 'Oz', 'O2']
    path = r'C:/Users/tomry/Documents/EEG/train/'
    series = range(start_series, end_series+1)
    subject_data = []
    for serie in series:
        data_filename = path + 'subj%d_series%d_%s.csv' % (subject, serie, filetype)
        data = pd.read_csv(data_filename)
        
        if filetype == 'data':
            data = data.loc[:, keep_channels]
        
        if filetype == 'events':
            data.drop('id', inplace=True, axis=1)
        
        subject_data.append(data)
    subj_data = pd.concat(subject_data)
    return subj_data.to_numpy()

def load_all_data(filetype, start_series, end_series):
    keep_channels =['Fp1', 'Fp2', 
                'F7', 'F3', 'Fz', 'F4', 'F8', 
                'FC5', 'FC6', 
                'T7', 'C3', 'C4', 'T8', 
                'CP5', 'CP6',
                'P7', 'P3', 'P4', 'P8', 
                'O1', 'Oz', 'O2']
    path = r'C:/Users/tomry/Documents/EEG/train/'
    series = range(start_series, end_series+1)
    subjects = range(1, 13)
    overall_data = []
    for subject in subjects:
        subject_data = []
        for serie in series:
            data_filename = path + 'subj%d_series%d_%s.csv' % (subject, serie, filetype)
            data = pd.read_csv(data_filename)
        
            if filetype == 'data':
                data = data.loc[:, keep_channels]
            
            if filetype == 'events':
                data.drop('id', inplace=True, axis=1)
        
            subject_data.append(data)
        subj_data = pd.concat(subject_data)
        overall_data.append(subj_data)
    all_data = pd.concat(overall_data)
    return all_data.to_numpy()
            
            
def time_series_gen(data, targets, time_series_len):
    ''' Creates batches of temporal data from data. '''
    return TimeseriesGenerator(data=data, targets=targets, 
                               length=time_series_len, 
                               batch_size=512)
              