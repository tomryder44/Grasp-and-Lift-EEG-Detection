
import os 
os.chdir(r'C:/Users/tomry/Documents/EEG')

import numpy as np
from classification.gal_classification import optimise_model
from preprocessing.gal_preprocessing import get_data, preprocess
from feature_extraction.gal_feature_extraction import compute_feature_space
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import plot_roc_curve

subject = 2
grid = np.logspace(-1.3, 1.3, 10)

x_train, x_val, x_test, y_train, y_val, y_test = get_data(subject)

x_train, x_val, x_test, y_train, y_val, y_test = preprocess(x_train, x_val, x_test,
                                                        y_train, y_val, y_test)

win_length = 0.5

x_train, x_val, x_test, y_train, y_val, y_test = compute_feature_space(
                                                        x_train, x_val, x_test, 
                                                        y_train, y_val, y_test,
                                                        win_length)

model = OneVsRestClassifier(LogisticRegression(max_iter=2000))

model, _ = optimise_model(x_train, x_val, y_train, y_val, 
                                           model, grid)

names = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 
         'Replace', 'BothReleased']
handstart_disp = plot_roc_curve(model.estimators_[0], x_test, y_test[:,0], name=names[0])
for i, estimator in enumerate(model.estimators_[1:], start=1):
    plot_roc_curve(estimator, x_test, y_test[:,i], ax=handstart_disp.ax_, name=names[i])
handstart_disp.figure_.suptitle('Subject %d ROC Curve' % subject)

