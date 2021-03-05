## Discussion

---
---

### Artifact removal 
Using ICA for artifact removal lowers the AUROC across all algorithms. For these results, only eyeblink artifacts were removed to prevent artifact misidentification and loss of useful information. One possible explanation is that the activity in the independent source containing blink artifacts contains useful, event-discriminant information. The artifacts are removed by setting the source to zero, but perhaps a thresholding technique is required to just remove the large spikes. 

### Signal filtering
Filtering the signals into the seperate brain rhythms improves performance for the longer windows of 1 and 2 seconds, but not for the windows of 0.25 and 0.5 seconds. One potential explanation for this is that the quality of the PSD estimate improves with longer windows, resulting in more useful features.

### Window length
Window lengths of 0.25, 0.5 and 1 second all perform well. A window length of 2 seconds results in a significant drop in performance, with all the algorithms using a window length of 2 at the bottom of the table. A logical explanation for this is that data from 2 seconds before a GAL event contains no/little useful information.  

### Dimensionality reduction with PCA
Algorithms using PCA see a decrease in AUROC, perhaps expected given the inherent loss of information. However, the reduction in number of features and in turn training time is significant. It can be seen PCA reduces dimensionality more than L1 regularisation. 

### Regularisation 
Algorithms using L2 regularisation perform better than their L1 counterpart. L1 regularisation prioritises a sparse solution, completely zeroing out coefficients. In contrast, L2 regularisation will shrink the coefficients of less useful features, but the features are still used, perhaps resulting in a slightly better AUROC. 
 
## Future Work:
- General:
  - Use best algorithm on Kaggle test set and upload 
  - More in-depth statistical analysis of results
  - Algorithm comparison tool
- Independent component analysis:
  - More extensive artifact removal 
  - Automated the artifact identification process
- Filtering:
  - Frequency sub-bands for the filter bank
- Feature Extraction:
  - Time-frequency domain features 
  - Covariance matrix as a feature 
  - Other dimensionality reduction methods e.g. sequential feature selection
- Classification 
  - Test different types of models (does the best algorithm for logistic regression translate to other models?)
  - Ensemble of classifiers
  - Postprocessing of model outputs
