---
title: Preprocessing
---

<span style="color:green"> Raw EEG signals are full of noise that can obscure or hide true brain activity in the signals, making the task of classification more difficult. The first step is therefore to process the raw EEG signals in an attempt to maximise signal-to-noise ratio. </span>

---
---

## Brain Rhythms
The main frequencies of the EEG signals change depending on the type of brain activity. Brain rhythms are frequency ranges corresponding to EEG signal frequencies exhibited during different types of activity. They are typically defined as delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz) and gamma (30-80 Hz). Broadly speaking, the lower frequency delta and theta bands correspond to sleep and relaxation states, whilst the higher frequency alpha, beta and gamma bands are exhibited during times of wakefullness. The image below shows some raw EEG data filtered into the different rhythms.

![subject 1 channel 1 rhythms](images/subj1_rhythms.png) 

## Time-Domain Filtering
The first preprocessing step is therefore the remove frequencies outside the range of the brain rhythms, based on the assumption that the useful information is contained within these frequencies. This is performed by bandpass filtering the signals between 0.5-80 Hz. Interestingly, while it might make sense to keep only the alpha, beta and gamma rhythms, it was found experimentally there is useful information contained in the delta and theta bands.

### Filter Implementation
The signals are filtered causally, i.e. using only present and past samples, to prevent data leakage from future samples. An inherent problem with causal filtering is that a time delay is introduced to the filtered signals (see filter_testing.ipynb for an example). This is an issue for supervised learning where each sample has a corresponding label at the same time. For minimal time delay, IIR filters are used to filter the EEG signals, despite the non-linear phase causing some distortion of the signal within the passband frequencies. The filter coefficients for 4th order Butterworth filters are obtained using he SciPy `butter` functon and the signals causally filtered using `lfilter`.

## Decimation
To reduce the computational costs of feature extraction and model training, the signals are downsampled. Because downsampling effectively lowers the sampling rate, low-pass filtering is necessary to remove higher frequency components (above half the new sampling rate) that would cause aliasing. As above, the signals are bandpass filtered with a higher frequency cutoff of 80 Hz. The sampling frequency must therefore be at least 160 Hz to avoid aliasing. This allows for downsampling by taking every 3rd sample (500/160=3.125), resulting in a new sampling frequency of 166.6 Hz.

## Artifact Removal
EEG signals are contaminated with artifacts, signals not produced by brain activity but picked up in the EEG recordings. These artifacts often have a greater amplitude than the signals generated in the brain and can therefore obscure the actual brain activity. Common artifacts found in EEG signals arise from both physiological/internal (e.g. blinking (EOG), heart beat (ECG), muscle activity (EMG) and breathing) and non-physiological/external (e.g. mains interference, cable movement) sources. 

To best classify EEG signals, the artifacts should be removed, leaving only the signals produced directly from brain activity. In reality, perfect separation between artifact and brain activity from EEG signals is not possible, and so removal of artifacts from EEG signals inadvertently results in the loss of useful EEG information. 

### Filtering
Bandpass filtering the signals removes artifacts outside the 0.5-80 Hz range. This includes DC offset, very low frequency artifacts such as breathing, and EMG signals above 80 Hz. As shown in filter_testing.ipynb, mains interference at 50 Hz has already been removed.

### Independent Component Analysis
Another artifact removal technique is necessary to remove those present in the 0.5-80 Hz range. Independent component analysis (ICA) is one such technique commonly used to remove artifacts from EEG signals. ICA decomposes the 32 channels of EEG data into 32 maximally independent signals/sources, which can isolate artifacts hidden in the EEG data, allowing for their removal. 

ICA is implemented with the `FastICA` algorithm in scikit-learn. For each subject, an unmixing matrix is computed to transform the EEG signals into the independent sources. The sources are not ordered in any way, so identification of artifacts requires manual (or automated) inspection. In this work, the sources are presented in runtime, for example the sources of subject 1 are:

![subj1_ica](images/subj1_ica.png)

Using a guide such as https://labeling.ucsd.edu/tutorial/labels , any artifacts can be identified. For example, eyeblink artifacts can be seen in the source 4th from the bottom, with their distinctive large peaks. To reject these eyeblinks, the 28th channel of independent sources is set to zero. When the inverse transformation is applied back to the mixed EEG signals, the eyeblinks are removed, as shown below:

![subj1_w_eyeblink](images/subj1_eyeblink.png)

## Filter Bank
The use of a filter bank is investigated. The EEG signals are passed through five bandpass filters with cutoff frequencies corresponding to the brain rhythms defined above, and concatenated column-wise. This increases dimensionality of the data from 32 channels to 160. The idea behind using a filter bank like this is that because features are extracted from specific rhythms rather than the entire frequency spectrum, there is more discriminatory information about the different events available.  

![filter_bank_fig](images/filter_bank.png)