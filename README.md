# Grasp-and-Lift EEG Detection
This repository contains the code used for research into the detection of events during grasp-and-lift trials using EEG signals, based on the Grasp-and-Lift EEG Detection Kaggle competition https://www.kaggle.com/c/grasp-and-lift-eeg-detection . This README contains the methodology used and results obtained. 

## Table of Contents
[Project Overview](#project-overview) 
- [Task](#task)
- [Grasp-and-Lift](#grasp-and-lift)
- [EEG Data](#the-eeg-data)
- [Predictions and Evaluation](#predictions-and-evaluation)
- [Approach](#approach)

[Preprocessing](#preprocessing)
- [Time-Domain Filtering](#Time-domain-Filtering)
  - [Filter Implementation](#Filter-Implementation)
- [Decimation](#Decimation)
- [Artifact Removal](#Artifact-Removal)
  - [Filtering](#Filtering)
  - [Independent Component Analysis](#Independent-Component-Analysis)
- [Filter Bank](#Filter-Bank)

[Feature Extraction](#Feature-Extraction)
- [Epoching](#epoching)
- [Time-Domain Features](#Time-Domain-Features)
- [Frequency-Domain Features](#Frequency-Domain-Features)
- [Principal Component Analysis](#Principal-Component-Analysis)

[Classification](#classification)
- [Subject-Specific Models](#Subject-Specific-Models)
- [Regularisation](#Regularisation)
  - [Grid Search](#Grid-Search) 

[Results](#Results)

[Discussion](#Discussion)
- [Future Work](#Future-Work)

## Project Overview
#### Task
The aim of this work is to research how different signal processing and machine learning techniques and algorithm parameters impact the prediction of hand movements during a 'grasp-and-lift' (GAL) motion, using the recorded electroencephalogram (EEG) signals. Being able to accurately extract intent from raw EEG signals opens up many possibilities of developing healthcare technologies to improve the quality of life of amputees and those with neurological disorders. In the case of predicting the events of a GAL, the predictions could be fed to a robotic prosthesis, allowing for intuitive control for upper-limb amputees. 

#### Grasp-and-Lift
The GAL is defined as 6 distinct events, taking place in the same order, as:
1. HandStart - reach for small object
2. FirstDigitTouch - grasp object with index finger and thumb
3. BothStartLoadPhase - lift the object a few centimetres
4. LiftOff - hold it in the air for a few seconds
5. Replace - place the object back down
6. BothReleased - release the object 

#### The EEG Data
The dataset downloaded from Kaggle consists of the EEG recordings for 12 different subjects. Each subject has 10 series of recordings, each of which contains 30 GALs. 32 channels of EEG data are recorded, with a sampling rate of 500 Hz. Further information about the data acquisition can be found at: https://www.nature.com/articles/sdata201447 .

Each sample of the EEG data has a corresponding one-hot encoded events vector, with one column for each type of event. A 1 in a column means the event has occurred within +-150 ms of that sample. The image below shows the first 20 seconds of subject 1's data from channel 1, and the corresponding events at each sample. The events overlap, making this a multi-label classificaton problem (can be more than one event at a time).

![subject 1 channel 1 EEG plot](images/subj1_channel1_plot.png)

#### Predictions and Evaluation
The evaluation metric used in the Kaggle competition is the area under the receiver operating characteristic curve (AUROC), averaged across all events. The ROC curve plots the true positive rate vs the false positive rate for different decision probability thresholds (the probability at which the class decision becomes 1). A classifier with a threshold = 0.1 is going to make far more positive (i.e. 1) decisions than a classifier with threshold = 0.9, but at the cost of specificity. Using the AUROC allows for the evaluation of predictions without having to decide on the decision probability threshold. Given this metric, the *probability* of each GAL event happening at each sample is predicted, not the 1 or 0 class label.  

#### Approach
Scores of 0.97+ have been achieved on the Kaggle test set using deep learning model ensembles. This is a very computationally intensive approach that requires GPUs for realistic training times. This work focuses on less intensive techniques, firstly investigating how preprocessing and feature extraction techniques can improve classification using only logistic regression. Based on the findings, more powerful models will be tested in the future.

A note on testing: the Kaggle submission restriction of 4 per day prevents testing a large number of different algorithms. For this reason, the test set used is the 7th and 8th series of the training data, for which the events are available to compute the AUROC score.

## Preprocessing
Raw EEG signals contain noise that reduces the quality of extracted features and in turn classification performance. The EEG signals are therefore processed in an attempt to maximise signal-to-noise ratio. 

#### Time-Domain Filtering
The main frequencies of the EEG signals change depending on the type of brain activity. Brain rhythms are frequency ranges corresponding to EEG signal frequencies exhibited during different types of activity. They are typically defined as delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz) and gamma (30-80 Hz). Broadly speaking, the lower frequency delta and theta bands correspond to sleep and relaxation states, whilst the higher frequency alpha, beta and gamma bands are exhibited during times of wakefullness. It would make sense therefore, to keep only the alpha, beta and gamma rhythms. It was found experimentally however that there is useful information at lower frequencies that improves predictions. 


With the 500 Hz sampling frequency, the EEG signals contain frequencies up to 250 Hz. To keep only the frequencies within these defined rhythms, the EEG signals are bandpass filtered between 0.5-80 Hz. The EEG data of subject 1 channel 1, filtered into the different rhythms is shown below. 

![subject 1 channel 1 rhythms](images/subj1_rhythms.png) 

##### Filter Implementation
The signals are filtered causally, i.e. using only present and past samples, to prevent data leakage from future samples. An inherent problem with causal filtering is that a time delay is introduced to the filtered signals (see filter_testing.ipynb for an example). This is an issue for supervised learning where each sample has a corresponding label at the same time. For minimal time delay, IIR filters are used to filter the EEG signals, despite the non-linear phase causing some distortion of the signal within the passband frequencies. The filter coefficients for 4th order Butterworth filters are obtained using he SciPy `butter` functon and the signals causally filtered using `lfilter`.

#### Decimation
To reduce the computational costs of feature extraction and model training, the signals are downsampled. Because downsampling effectively lowers the sampling rate, low-pass filtering is necessary to remove higher frequency components (above half the new sampling rate) that would cause aliasing. As above, the signals are bandpass filtered with a higher frequency cutoff of 80 Hz. The sampling frequency must therefore be at least 160 Hz to avoid aliasing. This allows for downsampling by taking every 3rd sample (500/160=3.125), resulting in a new sampling frequency of 166.6 Hz.

#### Artifact Removal
EEG signals are contaminated with artifacts, signals not produced by brain activity but picked up in the EEG recordings. These artifacts often have a greater amplitude than the signals generated in the brain and can therefore obscure the actual brain activity. Common artifacts found in EEG signals arise from both physiological/internal (e.g. blinking (EOG), heart beat (ECG), muscle activity (EMG) and breathing) and non-physiological/external (e.g. mains interference, cable movement) sources. 

To best classify EEG signals, the artifacts should be removed, leaving only the signals produced directly from brain activity. In reality, perfect separation between artifact and brain activity from EEG signals is not possible, and so removal of artifacts from EEG signals inadvertently results in the loss of useful EEG information. 

##### Filtering
Bandpass filtering the signals removes artifacts outside the 0.5-80 Hz range. This includes DC offset, very low frequency artifacts such as breathing, and EMG signals above 80 Hz. As shown in filter_testing.ipynb, mains interference at 50 Hz has already been removed.

##### Independent Component Analysis
Another artifact removal technique is necessary to remove those present in the 0.5-80 Hz range. Independent component analysis (ICA) is one such technique commonly used to remove artifacts from EEG signals. ICA decomposes the 32 channels of EEG data into 32 maximally independent signals/sources, which can isolate artifacts hidden in the EEG data, allowing for their removal. 

ICA is implemented with the `FastICA` algorithm in scikit-learn. For each subject, an unmixing matrix is computed to transform the EEG signals into the independent sources. The sources are not ordered in any way, so identification of artifacts requires manual (or automated) inspection. In this work, the sources are presented in runtime, for example the sources of subject 1 are:

![subj1_ica](images/subj1_ica.png)

Using a guide such as https://labeling.ucsd.edu/tutorial/labels , any artifacts can be identified. For example, eyeblink artifacts can be seen in the source 4th from the bottom, with their distinctive large peaks. To reject these eyeblinks, the 28th channel of independent sources is set to zero. When the inverse transformation is applied back to the mixed EEG signals, the eyeblinks are removed, as shown below:

![subj1_w_eyeblink](images/subj1_eyeblink.png)

#### Filter Bank
The use of a filter bank is investigated. The EEG signals are passed through five bandpass filters with cutoff frequencies corresponding to the brain rhythms defined above, and concatenated column-wise. This increases dimensionality of the data from 32 channels to 160. The idea behind using a filter bank like this is that because features are extracted from specific rhythms rather than the entire frequency spectrum, there is more discriminatory information about the different events available.  

## Feature Extraction
Features are extracted from the EEG signals that aim to capture the important, event-discriminatory, information. This allows for reduction in the size of the data set without the loss of information, resulting in shorter training times and potentially better classification performance. 

#### Epoching
The signals are split into segments of data known as epochs, from which a set of features are extracted. A sliding window is used to pass through the data, extracting features as it goes. The length of the window, dictating how much previous information is useful for predictions, is an important parameter affecting classification performance. Window lengths of 0.25, 0.5, 1 and 2 seconds are tested, all with 50% overlap. The set of features extracted from a window take the events label of the final sample in that window.

#### Time-Domain Features
The first type of feature extracted from each epoch are temporal features, describing the EEG signal amplitude over time. Simple statistical properties - maximum, minimum and mean absolute value, and Hjorth parameters - activity, mobility, complexity, are computed. 

#### Frequency-Domain Features
Spectral features are also extracted from the signals. For each epoch the power spectral density (PSD) estimate, describing the power of the signal at different frequencies, is computed. Welch's PSD estimate computes the modified (window function to reduce spectral leakage) periodogram across overlapping segments of the EEG signals, reducing the variance of the estimate over the periodogram and Bartlett's method (non-overlapping segments). Welch's PSD estimate is computed with SciPy's `welch` function, with 50% overlap between segments and a Hann window function. The maximum power and the frequency at which maximum power occurs are extracted from each PSD, in addition to the average power across each quarter of the PSD (average across each quarter used rather than the PSD itself to reduce number of features). 

#### Principal Component Analysis
Extraction of multiple features from multiple channels of EEG data results in a highly dimensional feature space, inevitably containing redundant features that can worsen classification performance and increase training times. It can therefore be useful to reduce the dimensonality of the feature space by keeping the useful information whilst throwing out the redundant information. 

Principal component analysis (PCA) is one such technique that can be used for dimensionality reduction. In PCA, orthonormal vectors (the principal components (PCs)) are computed, with the first one being the direction in the feature space with maximal variance. The second PC is therefore the vector with maximal variance that is also orthogonal to the first PC and so on for n PCs for an n-dimensional feature space. By selecting a number of PCs that explain some large proportion of variance in the feature space, and dropping those that account for little variance, the dimensionality of the feature space is reduced when transformed with the kept PCs. PCA is implemented with scikit-learn, and the number of features kept is set as `n_components=0.95`, using the number of components that cumulatively explain 95% of the variance in the feature space. PCA is applied after normalisation as it is sensitive to scale of features. 

## Classification
Logistic regression, a binary classifier, is extended to multiclass and multilabel classification by constructing a *one-vs-rest* classifier, in which a logistic regression model is trained for each class, where the remaining 5 classes are considered the other class. For example, a model is trained to predict *HandStart* or *not HandStart*, and another is trained to predict *FirstDigitTouch* or *not FirstDigitTouch*. This is implemented in scikit-learn with the `OneVsRestClassifier` wrapper that contains the 6 models. 

#### Subject-Specific Models
The characteristics of the EEG signals in both time and frequency domain vary between different subjects. Therefore, a classifier trained on one subject's features won't perform well on another subject's features. For this reason, a one-vs-rest classifier is trained for **each** subject, resulting in 12 classifiers per algorithm. On a side note, subject-independent models have been successful using deep-learning approaches, where using data from different subjects serves as a type of regularisation. For real-life use, this would have the advantage of reducing or foregoing a calibration period for a user. 

#### Regularisation
When the coefficients of a logistic regression model get too large, the model becomes too complex and can't generalise well to unseen data, an issue known as overfitting. Regularisation is used to reduce overfitting during training by penalising large weight values through the use of a penalty term added to the loss function. Both L1 and L2 regularisation are used:
- L1 penalty - the sum of the absolute value of the weights is added to the loss function. L1 regularisation acts as a type of feature selection as the coefficients of unimportant features are often shrank to zero rather than just reduced. 
- L2 penalty - the sum of the squared value of weights is added to the loss function.

##### Grid Search
A coefficient before the penalty terms control the strength of regularisation. If the coefficient is too small, the model becomes too complex and overfits the data, and vice versa with a coefficient that is too large. It's therefore an important hyperparameter that needs to be optimised. A grid search is used to find the optimal penalty coefficient by testing on a holdout validation set (70-30 training-validation split) and choosing the coefficient corresponding the largest AUROC. 

## Results
Summary of what has been being tested:
- Single 0.5-80 Hz bandpass filter vs filter bank approach
- No ICA vs eyeblink artifact removal with ICA
- Feature extraction window length - 0.25, 0.5, 1 and 2 seconds
- No PCA vs PCA for dimensionality reduction
- L1 vs L2 penalties in logistic regression

**64 algorithms tested in total**

Note that all three metrics - training time, number of features and the AUROC score are all averaged across all subjects. 

| ICA   | Filter      |   Window length | PCA   | Penalty   |   Train time |   No. feats |   AUROC |
|-------|-------------|-----------------|-------|-----------|--------------|-------------|---------|
| False | bandpass    |            0.5  | False | l2        |        8.048 |     291     |   0.798 |
| False | bandpass    |            0.25 | False | l2        |       18.447 |     291     |   0.788 |
| False | bandpass    |            0.5  | False | l1        |       16.852 |     275.625 |   0.783 |
| False | filter bank |            1    | False | l2        |        7.422 |    1443     |   0.783 |
| False | filter bank |            0.5  | False | l2        |       24.936 |    1443     |   0.782 |
| False | filter bank |            0.25 | False | l2        |       54.911 |    1443     |   0.779 |
| False | bandpass    |            1    | False | l2        |        3.621 |     291     |   0.778 |
| False | bandpass    |            0.25 | False | l1        |       35.112 |     279.819 |   0.778 |
| False | filter bank |            1    | True  | l2        |        0.465 |     398.583 |   0.776 |
| False | filter bank |            1    | False | l1        |       48.758 |    1063.75  |   0.774 |
| False | filter bank |            0.5  | False | l1        |       96.66  |    1264.1   |   0.774 |
| False | filter bank |            0.25 | True  | l2        |        3.298 |     396.833 |   0.772 |
| True  | bandpass    |            0.5  | False | l2        |        7.716 |     291     |   0.772 |
| False | filter bank |            0.25 | False | l1        |      201.602 |    1011.96  |   0.772 |
| False | filter bank |            0.5  | True  | l2        |        1.165 |     315.917 |   0.77  |
| False | filter bank |            1    | True  | l1        |        9.621 |     264.208 |   0.769 |
| False | bandpass    |            0.5  | True  | l2        |        0.592 |      80.917 |   0.769 |
| False | filter bank |            0.25 | True  | l1        |       44.339 |     337.139 |   0.766 |
| True  | bandpass    |            0.25 | False | l2        |       17.19  |     291     |   0.766 |
| False | filter bank |            0.5  | True  | l1        |       15.392 |     252.944 |   0.765 |
| False | bandpass    |            0.5  | True  | l1        |        4.333 |      76.389 |   0.764 |
| False | bandpass    |            1    | False | l1        |        8.453 |     274.319 |   0.762 |
| True  | bandpass    |            0.5  | False | l1        |       16.723 |     279.042 |   0.758 |
| True  | bandpass    |            1    | False | l2        |        2.793 |     291     |   0.758 |
| False | bandpass    |            0.25 | True  | l2        |        0.944 |      87.167 |   0.755 |
| True  | filter bank |            0.5  | False | l2        |       22.737 |    1443     |   0.755 |
| True  | bandpass    |            0.5  | True  | l2        |        0.688 |      95.25  |   0.755 |
| True  | bandpass    |            0.25 | False | l1        |       34.79  |     270.986 |   0.755 |
| True  | filter bank |            0.25 | False | l2        |       60.028 |    1443     |   0.753 |
| False | bandpass    |            0.25 | True  | l1        |        7.45  |      84.736 |   0.752 |
| True  | filter bank |            1    | False | l2        |        7.336 |    1443     |   0.751 |
| True  | bandpass    |            0.5  | True  | l1        |        4.852 |      87.278 |   0.749 |
| False | bandpass    |            1    | True  | l2        |        0.256 |      76.917 |   0.748 |
| True  | filter bank |            0.5  | False | l1        |       96.982 |    1121.1   |   0.746 |
| True  | bandpass    |            1    | False | l1        |        9.098 |     273.958 |   0.746 |
| True  | filter bank |            0.25 | False | l1        |      197.119 |     995.139 |   0.745 |
| True  | filter bank |            1    | True  | l2        |        0.41  |     417.417 |   0.744 |
| True  | bandpass    |            0.25 | True  | l2        |        1.307 |      99.333 |   0.744 |
| True  | filter bank |            0.25 | True  | l2        |        2.642 |     414.5   |   0.744 |
| True  | filter bank |            1    | False | l1        |       51.189 |    1190.67  |   0.743 |
| True  | filter bank |            0.5  | True  | l2        |        1.058 |     326.583 |   0.743 |
| True  | bandpass    |            1    | True  | l2        |        0.26  |      92.583 |   0.742 |
| False | bandpass    |            1    | True  | l1        |        1.95  |      72.181 |   0.742 |
| True  | bandpass    |            0.25 | True  | l1        |        8.721 |      93.153 |   0.741 |
| True  | filter bank |            0.5  | True  | l1        |       17.675 |     253.5   |   0.736 |
| True  | filter bank |            0.25 | True  | l1        |       44.99  |     341.972 |   0.736 |
| True  | bandpass    |            1    | True  | l1        |        2.727 |      76.944 |   0.734 |
| True  | filter bank |            1    | True  | l1        |       10.811 |     255.583 |   0.733 |
| False | filter bank |            2    | False | l2        |        3.889 |    1443     |   0.719 |
| False | filter bank |            2    | True  | l2        |        0.258 |     344.333 |   0.717 |
| False | filter bank |            2    | False | l1        |       27.897 |    1173.81  |   0.716 |
| False | filter bank |            2    | True  | l1        |        4.864 |     236.389 |   0.706 |
| True  | filter bank |            2    | False | l2        |        3.422 |    1443     |   0.701 |
| True  | filter bank |            2    | True  | l2        |        0.245 |     362.083 |   0.697 |
| True  | filter bank |            2    | False | l1        |       26.249 |    1099.14  |   0.69  |
| True  | filter bank |            2    | True  | l1        |        5.192 |     237.736 |   0.685 |
| False | bandpass    |            2    | False | l2        |        1.199 |     291     |   0.684 |
| True  | bandpass    |            2    | False | l2        |        1.402 |     291     |   0.679 |
| False | bandpass    |            2    | True  | l2        |        0.139 |      66     |   0.673 |
| False | bandpass    |            2    | True  | l1        |        0.879 |      49.167 |   0.669 |
| True  | bandpass    |            2    | True  | l2        |        0.175 |      80.917 |   0.668 |
| True  | bandpass    |            2    | True  | l1        |        1.034 |      53.014 |   0.644 |
| False | bandpass    |            2    | False | l1        |        3.578 |     146.181 |   0.611 |
| True  | bandpass    |            2    | False | l1        |        2.948 |     118.875 |   0.6   |

Total run time: ~ 1 day with i5-8250U processor, 4x1.60GHz Cores, 16 GB RAM

Below shows the ROC curve for the best algorithm for subjects 1 and 2. 
![subj1_roc](images/subj1_roc.png)
![subj2_roc](images/subj2_roc.png)

## Discussion
- **Artifact removal**: Using ICA for artifact removal lowers the AUROC across all algorithms. For these results, only eyeblink artifacts were removed to prevent artifact misidentification and loss of useful information. One possible explanation is that the activity in the independent source containing blink artifacts contains useful, event-discriminant information. The artifacts are removed by setting the source to zero, but perhaps a thresholding technique is required to just remove the large spikes. 
- **Signal filtering**: Filtering the signals into the seperate brain rhythms improves performance for the longer windows of 1 and 2 seconds, but not for the windows of 0.25 and 0.5 seconds. One potential explanation for this is that the quality of the PSD estimate improves with longer windows, resulting in more useful features.
- **Window length**: Window lengths of 0.25, 0.5 and 1 second all perform well. A window length of 2 seconds results in a significant drop in performance, with all the algorithms using a window length of 2 at the bottom of the table. A logical explanation for this is that data from 2 seconds before a GAL event contains no/little useful information.  
- **Dimensionality reduction with PCA**: Algorithms using PCA see a decrease in AUROC, perhaps expected given the inherent loss of information. However, the reduction in number of features and in turn training time is significant. It can be seen PCA reduces dimensionality more than L1 regularisation. 
- **Regularisation**: Algorithms using L2 regularisation perform better than their L1 counterpart. L1 regularisation prioritises a sparse solution, completely zeroing out coefficients. In contrast, L2 regularisation will shrink the coefficients of less useful features, but the features are still used, perhaps resulting in a slightly better AUROC. 
 
#### Future Work:
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
