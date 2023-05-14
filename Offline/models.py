"""Inference models"""

def simpleModel(frequency_6Hz_Left, frequency_10Hz_Right, f_6Hz_threshold=3.0, f_10Hz_threshold=8.0) -> list:
    """
    This function classify the frequency of the 6Hz(Left) and 10Hz(Right) signals by giving threshold
    
    parameters
    ----------
    frequency_6Hz_Left <class 'list' or 'numpy.ndarray'>
    frequency_10Hz_Right <class 'list' or 'numpy.ndarray'>

    return
    ------
    outputs <class 'list'> ex. [5, 2, 2, 5, 2, ...]
    """
    if not isinstance(frequency_6Hz_Left, list):
        frequency_6Hz_Left, frequency_10Hz_Right = [frequency_6Hz_Left], [frequency_10Hz_Right]
    
    outputs = []

    for f6, f10 in zip(frequency_6Hz_Left, frequency_10Hz_Right):

        if f10 >= f_10Hz_threshold: outputs.append(5)
        elif f6 >= f_6Hz_threshold: outputs.append(2)
    
    return outputs