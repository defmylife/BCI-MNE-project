# 🧠Offline Analysis

This folder contains the following files :

1. **`utilities.py`**: This file stores functions that can be used for offline analysis, such as reading .XDF files, plotting visualizations, and conducting various analyses.

2. **`tutorial.ipynb`**: This Jupyter Notebook provides explanations for each function present in `utilities.py`. It serves as a guide to help you understand how each function works and provides instructions for customizing them according to your specific needs.

3. **`models.py`**: This file contains inference models that are utilized in the decoding pipeline.

4. **`trainModel.ipynb`**: In this Jupyter Notebook, you can find code for reading epoch data stored in a CSV file, performing pre-processing steps, and training a model. It provides a workflow for training a model based on the processed data. - *In progress*



## Installation

Install with pip

```bash
pip install mne mnelab pyxdf matplotlib pandas numpy tensorflow sklearn pyriemann
```
    



## Usage/Examples

Table of content :

1. [Reading XDF file](https://github.com/defmylife/BCI-MNE-project/tree/main/Offline#1-reading-xdf-file)
2. [Offline analysis (Epoching)](https://github.com/defmylife/BCI-MNE-project/tree/main/Offline#2-offline-analysis-(Epoching))
3. [Decoding and Result](https://github.com/defmylife/BCI-MNE-project/tree/main/Offline#3-decoding-and-result)


### 1. Reading XDF file

```python
from utilities import read_xdf

filename = 'Pipo_1_5_test1.xdf'

# Loading XDF file into raw: MNE-RawArray
raw = read_xdf(filename, 

    bandpass=(3.0, 15.0), # Bandpass 3Hz - 15Hz (default 0Hz - 45Hz)

    show_plot=False, 
    # show_plot : If True, show all EEG channels and able to zoom in-out, scaling

    show_psd=False,
    # show_psd : If True, show overall average power spectral density
)
```

#### show_plot
```python
raw = read_xdf(
    ...
    show_plot=True, 
    # show_plot : If True, show all EEG channels and able to zoom in-out, scaling
    ...
)
```
![Preview](preview/show_plot.png)

<!-- #### show_psd
```python
raw = read_xdf(
    ...
    show_psd=True, 
    # show_psd : If True, show overall average power spectral density
    ...
)
```
![Preview](preview/show_psd.png) -->



### 2. Offline analysis (Epoching)

```python
# Epoching, showing Power spectral density (PSD) split by Left-Right stimuli event
epochs = epoching(raw, filename,

    show_eeg=False,
    # show_eeg : If True, show all EEG channels and able to zoom in-out, scaling split by Left-Right stimuli

    show_psd=False,
    # show_psd : If True, show overall average power spectral density split by Left-Right stimuli

    show_time_freq=False,
    # show_time_freq : If True, show Time-Frequency plot split by Left-Right stimuli and each O1, Oz, O2, POz, Pz
)
```

#### show_psd
```python
epochs = epoching(
    ...
    show_psd=True,
    # show_psd : If True, show overall average power spectral density split by Left-Right stimuli
    ...
)
```
![Preview](preview/show_epoch.png)

<!-- #### show_eeg
```python
epochs = epoching(
    ...
    show_eeg=True, 
    # show_eeg : If True, show all EEG channels and able to zoom in-out, scaling split by Left-Right stimuli
    ...
)
```
![Preview](preview/show_eeg_L.png)
![Preview](preview/show_eeg_R.png) -->

#### show_time_freq
```python
epochs = epoching(
    ...
    show_time_freq=True, 
    # show_time_freq : If True, show Time-Frequency plot split by Left-Right stimuli and each O1, Oz, O2, POz, Pz
    ...
)
```
![Preview](preview/show_time_freq.png)



### 3. Decoding and Result
```python
# Decoding
outputs = decoding(epochs,

    plot=False,
    # plot    : If True, visualize plot all events, compare the two ranges of frequencies, and view the outputs.

    verbose=True,
    # verbose : If True, print the outputs and classification report in the terminal.
)
```

#### verbose
```python
outputs = decoding(
    ...
    verbose=True,
    # verbose : If True, print the outputs and classification report in the terminal.
    ...
)
```
```
Labels      : [5, 2, 5, 2, 2, 5, 2, 5, 2, 5, 5, 2, 5, 2, 5, 2]
Predictions : [5, 2, 5, 2, 2, 5, 2, 5, 2, 5, 5, 2, 5, 2, 5, 2]

              precision    recall  f1-score   support

           2       1.00      1.00      1.00         8
           5       1.00      1.00      1.00         8

    accuracy                           1.00        16
   macro avg       1.00      1.00      1.00        16
weighted avg       1.00      1.00      1.00        16
```





