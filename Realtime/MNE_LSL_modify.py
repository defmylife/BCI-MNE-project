import matplotlib.pyplot as plt

from mne.datasets import sample
from mne.io import read_raw_fif

from mne_realtime import LSLClient, MockLSLStream

import numpy as np
import math
import pylsl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from typing import List

print(__doc__)

# Basic parameters for the plotting window
plot_duration = 5  # how many seconds of data to show
update_interval = 60  # ms between screen updates
pull_interval = 500  # ms between each pull operation

# this is the host id that identifies your stream on LSL
host = 'openbcigui'
# this is the max wait time in seconds until client connection
wait_max = 5
n_epochs = 5

def main():
    # Create the pyqtgraph window
    pw = pg.plot(title='LSL Plot')
    pltpy = pw.getPlotItem()
    pltpy.enableAutoRange(x=False, y=True)
    
    # MNE-LSL Client
    lsl_client = LSLClient(info=None, host=host, wait_max=wait_max)
    
    def epoch_plot():
        with LSLClient(info=None, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq']) 
            
            # let's observe ten seconds of data
            # for ii in range(n_epochs):
            #     print('Got epoch %d/%d' % (ii + 1, n_epochs))
            #     plt.cla()
            #     epoch = client.get_data_as_epoch(n_samples=sfreq)
            #     epoch.average().plot(axes=ax)
            #     plt.pause(1.)
            epoch = client.get_data_as_epoch(n_samples=sfreq)
            epoch.average().plot(axes=ax)
            plt.draw()
    
    _, ax = plt.subplots(1)
        
    # MNE-LSL Client
    # with LSLClient(info=None, host=host, wait_max=wait_max) as client:
    #     client_info = client.get_measurement_info()
    #     sfreq = int(client_info['sfreq']) 
        
    #     # let's observe ten seconds of data
    #     for ii in range(n_epochs):
    #         print('Got epoch %d/%d' % (ii + 1, n_epochs))
    #         plt.cla()
    #         epoch = client.get_data_as_epoch(n_samples=sfreq)
    #         epoch.average().plot(axes=ax)
    #         plt.pause(1.)
    #     plt.draw()

    def scroll():
        """Move the view so the data appears to scroll"""
        # We show data only up to a timepoint shortly before the current time
        # so new data doesn't suddenly appear in the middle of the plot
        fudge_factor = pull_interval * .002
        plot_time = pylsl.local_clock()
        pw.setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)

    def update():
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        mintime = pylsl.local_clock() - plot_duration
        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.
        # for inlet in inlets:
        #     inlet.pull_and_plot(mintime, plt)
        epoch_plot()
        
    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    import sys

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QGuiApplication.instance().exec_()

if __name__ == '__main__':
    main()
