
import json

def save_json(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

ICA = [False, True]
filter_method = ['bandpass', 'filter bank']
window_lengths = [0.25 , 0.5, 1, 2]
PCA = [False, True]
reg_pens = ['l2', 'l1']

algorithm_id = 1

for do_ica in ICA:
    for filt in filter_method:
        for win in window_lengths:
            for do_pca in PCA:
                for pen in reg_pens:
                    algorithm = {}
                    algorithm['ICA'] = do_ica
                    algorithm['Filter'] = filt 
                    algorithm['Window length'] = win
                    algorithm['PCA'] = do_pca
                    algorithm['Penalty'] = pen
                    filename = 'algorithm_' + str(algorithm_id) + '.txt'
                    save_json(filename, algorithm)
                    algorithm_id += 1
