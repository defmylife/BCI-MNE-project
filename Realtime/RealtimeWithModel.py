"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
from scipy import signal
import numpy as np
from models import CNNModel

def main():
    # resolve EEG stream
    print("looking for an EEG stream...")
    eeg_streams = resolve_stream('type', 'EEG')
    
    # create a new inlet to read from the streams
    print("creating inlets...")
    eeg_inlet = StreamInlet(eeg_streams[0])

    # FIR filter coefficients
    fs = 250 # sampling rate in Hz
    nyq = 0.5 * fs # Nyquist frequency
    # cutoff_freq = 25 # desired cutoff frequency in Hz
    low_edge = 2
    high_edge = 50
    # numtaps = 101 # number of filter taps

    # design the band pass filter
    nyq = 0.5 * fs
    f1_norm = low_edge / nyq
    f2_norm = high_edge / nyq
    b, a = signal.butter(4, [f1_norm, f2_norm], 'bandpass', analog=False)
    zi = signal.lfilter_zi(b,a)


    # low pass
    # normalized_cutoff_freq = cutoff_freq / nyq
    # b = signal.firwin(numtaps, normalized_cutoff_freq, window='hamming')
    # zi = signal.lfilter_zi(b, 1)

    # normalize
    # min_val = 0.0
    # max_val = 0.0

    time_count = 0.0
    epoch_5s = np.zeros(shape=(5,1250))
    temp = []

    CNN = CNNModel()
    CNN.load_weights('CNN.h5')

    while True:
        eeg_sample, timestamps = eeg_inlet.pull_sample()

        if timestamps:
            time_count += 1 * 0.004
            time_count = np.round(time_count ,3)

            filtered_chunk, zi = signal.lfilter(b, 1, eeg_sample, zi=zi)
            filtered_chunk = [filtered_chunk[1], filtered_chunk[2], filtered_chunk[3],filtered_chunk[2],filtered_chunk[5]]

            temp.append(filtered_chunk)
                
            if time_count == 1.0:
                temp = np.asanyarray(temp)
                print('every 1s', np.shape(temp.T))
                epoch_5s = np.delete(epoch_5s, slice(0,250), axis =1)
                print('delete',epoch_5s.shape)
                epoch_5s = np.append(epoch_5s, temp.T, axis= 1)
                print('-----------------------------------------------------------------------------------')
                print(epoch_5s)
                print(epoch_5s.shape)

                ########send model#####
                out = CNN.model_predict_classes(np.reshape(epoch_5s, (1, 5, 1250)))
                predict_value = CNN.model_predict(np.reshape(epoch_5s, (1, 5, 1250)))
                print('predict class is', out, predict_value)

                temp = np.ndarray.tolist(temp)
                temp = []
                time_count = 0.0
                
                
            ##### Basic Normalize ######
            # if min(filtered_chunk) < min_val:
            #     min_val = min(filtered_chunk)
            # if max(filtered_chunk) > max_val:
            #     max_val = max(filtered_chunk)

            # for i in range (0,len(filtered_chunk)):
            #     filtered_chunk[i] = (filtered_chunk[i] - min_val) / (max_val - min_val)

        time_before = timestamps            

if __name__ == '__main__':
    main()

