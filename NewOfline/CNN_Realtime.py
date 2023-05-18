"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
from scipy import signal
import numpy as np
from models import CNNModel
from gripper import Gripper

def main():
    # resolve EEG stream
    print("looking for an EEG stream...")
    eeg_streams = resolve_stream('type', 'EEG')
    
    # create a new inlet to read from the streams
    print("creating inlets...")
    eeg_inlet = StreamInlet(eeg_streams[0])

    time_count = 0.0
    epoch_5s = np.zeros(shape=(5,1250))
    temp = []

    CNN = CNNModel()
    CNN.load_weights('CNNN_New.h5')

    # grip = Gripper()

    while True:
        eeg_sample, timestamps = eeg_inlet.pull_sample()

        if timestamps:

            time_count += 1 * 0.004
            time_count = np.round(time_count ,3)

            data = [eeg_sample[1],eeg_sample[2],eeg_sample[3],eeg_sample[4],eeg_sample[5]]
            temp.append(data)
       
            if time_count == 1.0:
                temp = np.asarray(temp)
                epoch_5s = np.delete(epoch_5s, slice(0,250), axis =1)
                epoch_5s = np.append(epoch_5s, temp.T, axis= 1)
                send_epoch = epoch_5s
                predict_raw_value, predict_value = CNN.model_predict_classes(np.reshape(send_epoch, (1, 5, 1250)))
                print('-----------------------------------------------------------------------------------')
                print(f'Predict val : 6 Hz Left [{predict_raw_value[0][0]*100.0:.2f} %] 10 Hz Right [{predict_raw_value[0][1]*100.0:.2f} %] Idle [{predict_raw_value[0][2]*100.0:.2f} %]')
                print(f'Predicted class: {predict_value}')
                
                temp = np.ndarray.tolist(temp)
                temp = []
                time_count = 0.0
          
if __name__ == '__main__':
    main()

