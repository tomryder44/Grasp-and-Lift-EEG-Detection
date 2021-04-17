---
title: Experiment 2
---

<span style="color:green"> A major barrier for widespread BCI technology is that, because the properties of EEG data varies between people, a model trained on one person will not perform well on someone elses data. Therefore, a subject-specific model is required, in turn requiring sufficient data from a person from repeated trials of different task. Ideally, a model, given enough data, could capture EEG properties that are invariant across different subjects, such that a new user could use a BCI technology with little to no calibration. This experiment compares subject-specific models vs a single subject-independent model trained on all subjects data. </span>

---
---

## LSTM
This experiment also explores the use of deep learning, a subset of machine learning in which models can automatically learn features during training directly from EEG data, meaning the feature extraction stage can be skipped. Specifically, this experiment will use an long short-term memory (LSTM) recurrent neural network (RNN). RNNs were designed for time-series problems as they can capture dependencies in sequential data, however they suffer from the vanishing-gradient problem in which error backpropagated through the network gets smaller and smaller, meaning it can't learn longer-term temporal dependencies. The LSTM was designed to overcome this issue through the use of . The LSTM networks used are all single-layered with 100 units, followed by a dropout layer with dropout rate = 0.2, followed by a dense layer of 6 units with sigmoid activation for multi-label classification. The models are trained with early stopping to further prevent overfitting. 

## Datasets
The only EEG preprocessing is decimation, through 0.5-30 Hz bandpass filter and downsampling taking every 8th sample (new sampling freq: 62.5 Hz), to reduce computational costs. For the subject-independent model, the subject datasets are combined into one. All datasets are normalised to aid learning. To feed data of the right format into the LSTM networks, the Keras TimeseriesGenerator is used to obtain batches of temporal data. An important factor is *length*, that is the number of time-steps in a single sequence, and is determined by how far back we believe the information is useful for predictions at current time (e.g. for grasp-and-lift, EEG data from 1 hour ago is irrelevant). Based on the results from the logistic regression experiment, 0.5 seconds of samples (=31) is used as the length. 

## Results
The subject-specific and subject-independent models were tested on each subject's hold out test set. The training time and AUROC are presented.

#### Subject-specific LSTMs:



#### Subject-independent LSTM:

|   Subject |   AUROC |
|-----------|---------|
|         1 |   0.763 |
|         2 |   0.764 |
|         3 |   0.547 |
|         4 |   0.723 |
|         5 |   0.601 |
|         6 |   0.701 |
|         7 |   0.843 |
|         8 |   0.745 |
|         9 |   0.754 |
|        10 |   0.758 |
|        11 |   0.799 |
|        12 |   0.78  |

Model training time: 7:42:45 (h:m:s)


## Discussion
