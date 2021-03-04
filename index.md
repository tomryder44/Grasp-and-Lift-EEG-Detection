---
title: Grasp-and-Lift EEG Detection
---

### Task
The aim of this work is to research how different signal processing and machine learning techniques and algorithm parameters impact the prediction of hand movements during a 'grasp-and-lift' (GAL) motion, using the recorded electroencephalogram (EEG) signals. Being able to accurately extract intent from raw EEG signals opens up many possibilities of developing healthcare technologies to improve the quality of life of amputees and those with neurological disorders. In the case of predicting the events of a GAL, the predictions could be fed to a robotic prosthesis, allowing for intuitive control for upper-limb amputees. 

### Grasp-and-Lift
The GAL is defined as 6 distinct events, taking place in the same order, as:
1. HandStart - reach for small object
2. FirstDigitTouch - grasp object with index finger and thumb
3. BothStartLoadPhase - lift the object a few centimetres
4. LiftOff - hold it in the air for a few seconds
5. Replace - place the object back down
6. BothReleased - release the object 

### The EEG Data
The dataset downloaded from Kaggle consists of the EEG recordings for 12 different subjects. Each subject has 10 series of recordings, each of which contains 30 GALs. 32 channels of EEG data are recorded, with a sampling rate of 500 Hz. Further information about the data acquisition can be found at: https://www.nature.com/articles/sdata201447 .

Each sample of the EEG data has a corresponding one-hot encoded events vector, with one column for each type of event. A 1 in a column means the event has occurred within +-150 ms of that sample. The image below shows the first 20 seconds of subject 1's data from channel 1, and the corresponding events at each sample. The events overlap, making this a multi-label classificaton problem (can be more than one event at a time).

![subject 1 channel 1 EEG plot](images/subj1_channel1_plot.png)

### Predictions and Evaluation
The evaluation metric used in the Kaggle competition is the area under the receiver operating characteristic curve (AUROC), averaged across all events. The ROC curve plots the true positive rate vs the false positive rate for different decision probability thresholds (the probability at which the class decision becomes 1). A classifier with a threshold = 0.1 is going to make far more positive (i.e. 1) decisions than a classifier with threshold = 0.9, but at the cost of specificity. Using the AUROC allows for the evaluation of predictions without having to decide on the decision probability threshold. Given this metric, the *probability* of each GAL event happening at each sample is predicted, not the 1 or 0 class label.  

### Approach
Scores of 0.97+ have been achieved on the Kaggle test set using deep learning model ensembles. This is a very computationally intensive approach that requires GPUs for realistic training times. This work focuses on less intensive techniques, firstly investigating how preprocessing and feature extraction techniques can improve classification using only logistic regression. Based on the findings, more powerful models will be tested in the future.

A note on testing: the Kaggle submission restriction of 4 per day prevents testing a large number of different algorithms. For this reason, the test set used is the 7th and 8th series of the training data, for which the events are available to compute the AUROC score.
