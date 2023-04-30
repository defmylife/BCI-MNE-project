import pyxdf
from pyxdf import match_streaminfos, resolve_streams

import mne
from mnelab.io.xdf import read_raw_xdf

import matplotlib.pyplot as plt
from pprint import pprint

def read_xdf(filename, show_plot=True, show_psd=True, verbose=False) -> mne.io.array.array.RawArray:
    """
    Loading XDF file into MNE-RawArray. MNE-Python does not support this file format out of the box, 
    but we can use the pyxdf package and MNELAB to import the data. 

    return: MNE RawArray
    """
    # Read xdf
    # https://github.com/cbrnr/bci_event_2021#erders-analysis-with-mne-python
    streams = resolve_streams(filename) # streams has 2 types: EEG, Markers
    if verbose: pprint(streams) 

    # Find stearms type EEG
    stream_id = match_streaminfos(streams, [{"type": "EEG"}])[0]
    raw = read_raw_xdf(filename, stream_ids=[stream_id])
    # print(raw.info['bads'])

    # Set channel types
    #   set_channel_types({CHANNEL_NAME : CHANNEL_TYPE}) 
    raw.set_channel_types({'obci_eeg1_0': 'eeg'})   # FP1
    raw.set_channel_types({'obci_eeg1_1': 'eeg'})   # O1
    raw.set_channel_types({'obci_eeg1_2': 'eeg'})   # Oz
    raw.set_channel_types({'obci_eeg1_3': 'eeg'})   # O2
    raw.set_channel_types({'obci_eeg1_4': 'eeg'})   # POz
    raw.set_channel_types({'obci_eeg1_5': 'eeg'})   # Pz
    raw.set_channel_types({'obci_eeg1_6': 'eeg'})   # none
    raw.set_channel_types({'obci_eeg1_7': 'eeg'})   # none

    show = False
    # Plot EEG graph
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot
    if show_plot:

        raw.plot(
            duration=15, 
            start=0, 
            scalings=169, # You may edit scalings value later
        ) #, n_channels=8, bad_color='red'
        show = True

    # https://mne.tools/stable/generated/mne.io.RawArray.html#mne.io.RawArray.compute_psd
    if show_psd:
        raw.compute_psd(
            fmax=60,
            picks=[
                'obci_eeg1_1',
                'obci_eeg1_2',
                'obci_eeg1_3',
                'obci_eeg1_4',
                'obci_eeg1_5',
            ],                      # pick by channel name
            # picks='eeg',          # pick by channel type
            ).plot()
        show = True
        
    if show: plt.show()

    return raw


if __name__=='__main__':

    # raw = read_xdf("example.xdf")
    # raw = read_xdf("test01_OpenBCI.xdf", show_plot=True, show_psd=True)

    raw = read_xdf("omz1.xdf", show_plot=True, show_psd=False)