# 🚀Online task

This folder contains the following files :

1. **`utilities.py`**: This file stores functions that can be used for online task, such as decoding (*from [Offline folder](https://github.com/defmylife/BCI-MNE-project/tree/main/Offline)*)

2. **`models.py`**: This file contains inference models that are utilized in the decoding pipeline.

3. *please fill...*



## Installation

Install with pip

```bash
pip install mne mne_realtime mnelab pylsl pyxdf pyqtgraph bsl matplotlib pandas numpy tensorflow sklearn
```
    



## Usage/Examples

Loading and Epoching data in realtime

```python
# PLEASE FILL YOUR CODE HERE

# epochs = 

```


**Decoding**

```python
outputs = decoding( epochs,

    verbose=True,
    # verbose : If True, print the outputs and classification report in the terminal.
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





