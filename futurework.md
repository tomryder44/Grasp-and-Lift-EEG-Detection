---
title: Future Work
---

<span style="color:green"> This is an ongoing project that will be updated continuously. Different techniques and parameters that will be tested in the future are presented below. </span>

---
---

### General to-do
  - Use best algorithm on Kaggle test set and upload 
  - Algorithm comparison tool for website

### Independent Component Analysis - current focus
  - The current ICA approach has been to compute the unmixing matrix on the train data, and use that to transform the test data and remove the same source channels that were removed following manual inspection of training data sources. However, the distribution of data changes over time and so ICs should be computed over time, thus requiring a more suitable real-time approach. This involves two main steps:
    - Write the real-time algorithm
    - Train a classifier / some classifiers for automated artifact detection

### Feature Extraction
  - Time-frequency domain features 
  - Covariance matrix as a feature (results in (n²+n)/2 features for n channels)
  - Other dimensionality reduction methods e.g. sequential feature selection

### Classification 
  - Test larger, more complex models
  - Ensemble of classifiers
  - Postprocessing of model outputs
