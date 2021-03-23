
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
  
def grid_search(x_train, x_val, y_train, y_val, model, grid):
    ''' Performs a grid search to find optimal value of C for logistic regression.
    To speed up, if AUROC doesn't increase after 2 iterations, grid search ends. '''
    best_auroc = 0
    not_better_count = 0
    for c in grid:
        model.set_params(estimator__C=c)
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_val)
        auroc = roc_auc_score(y_val, y_pred)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = c
            not_better_count = 0
        else:
            not_better_count += 1
        print('C = %.3f, AUROC = %.3f' % (c, auroc))
        if not_better_count == 2:
            print('No improvements for two steps, ending grid search early.')
            break
    return best_C
    
         
def optimise_model(x_train, x_val, y_train, y_val, penalty, grid):
    ''' Optimise model with a grid search, using hold out set for validation,
    and evaluating with AUROC. '''
    
    if penalty == 'l2':
            model = OneVsRestClassifier(LogisticRegression(max_iter=2000))  
        
    elif penalty == 'l1':
            model = OneVsRestClassifier(LogisticRegression(max_iter=2000,
                                                           penalty='l1',
                                                           solver='saga',
                                                           tol=0.01)) # keeps training time down
    
    # perform grid search
    best_C = grid_search(x_train, x_val, y_train, y_val, model, grid)
    
    # concatenate training and validation sets
    x = np.concatenate((x_train, x_val))
    y = np.concatenate((y_train, y_val))
    
    # train final model on all non-test data with optimal C    
    model.set_params(estimator__C=best_C)
    start = time.time()
    model.fit(x, y)
    end = time.time()
    train_time = round(end-start,2)
    
    return model, train_time


    
    
    