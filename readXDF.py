import pyxdf
from pyxdf import match_streaminfos, resolve_streams

import mne
from mnelab.io.xdf import read_raw_xdf

import matplotlib.pyplot as plt
from pprint import pprint

def read_xdf(filename, show_plot=True, verbose=False) -> mne.io.array.array.RawArray:
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

    # Plot EEG graph
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot
    if show_plot:

        raw.plot(
            duration=5, 
            start=4.5, 
            scalings=169,
        ) #, n_channels=8, bad_color='red'
        plt.show()

    return raw


if __name__=='__main__':

    # raw = read_xdf("example.xdf")
    raw = read_xdf("test01_OpenBCI.xdf", show_plot=True)