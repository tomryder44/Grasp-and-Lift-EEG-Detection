import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import ast

# have an alert when ica ready to be inspected
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz

def independent_component_analysis(x, ica=None, col=None):
    ''' Perform independent component analysis on x. On training data, ICA
    is fit, independent sources checked for artifact channels, and then removed. 
    On test data, data is transformed with fitted ICA and same channels removed. '''    
    if ica:
        if col:
            independent_sources = ica.transform(x)
            independent_sources[:, col] = 0
            x_clean = ica.inverse_transform(independent_sources)
            return x_clean
        else:
            return x

    else:
        ica = FastICA().fit(x)
        independent_sources = ica.transform(x)
        # plot channels for manual inspection
        independent_sources_df = pd.DataFrame(data=independent_sources)
        independent_sources_df.iloc[1000:3000,:].plot(subplots=True, figsize=(40,40))
        plt.show()
        independent_sources_df.iloc[1000:1500,:].plot(subplots=True, figsize=(40,40))
        plt.show()
        # get input for channels to delete
        winsound.Beep(freq, duration) # alert!
        col_str = input('Remove component: ')
        if col_str:
            col = ast.literal_eval(col_str)
            independent_sources[:, col] = 0
            x_clean = ica.inverse_transform(independent_sources)
            return x_clean, ica, col
        else:
            return x, ica, col_str
