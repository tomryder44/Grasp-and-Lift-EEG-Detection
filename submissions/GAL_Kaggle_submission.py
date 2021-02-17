
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# algorithm parameters:
win = 
dr = 
model = OneVsRestClassifier(LogisticRegression(max_iter=2000))


subjects = range(1, 13)

# load and preprocess all datasets
subject_datasets = []
for subject in subjects:
    x_train, x_test, y_train, y_test = load_data(subject, for_kaggle=True)
    x_train, x_test, y_train, y_test = preprocess(x_train, x_test, y_train, y_test)
    subject_datasets.append((x_train, x_test, y_train, y_test))







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