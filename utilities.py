import pyxdf
from pyxdf import match_streaminfos, resolve_streams

import mne
from mnelab.io.xdf import read_raw_xdf

import matplotlib.pyplot as plt
from pprint import pprint


def read_xdf(filename, show_plot=True, show_psd=True, verbose=False, plot_scale=169) -> mne.io.array.array.RawArray:
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
            scalings=plot_scale, # You may edit scalings value later
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


def show_epoch(raw, filename=None, show_eeg=False, plot_scale=169):
    raw_eeg = raw.pick_channels([
                    'obci_eeg1_1',
                    'obci_eeg1_2',
                    'obci_eeg1_3',
                    'obci_eeg1_4',
                    'obci_eeg1_5',
                ])

    events, event_dict = mne.events_from_annotations(raw_eeg)

    epochs = mne.Epochs(raw_eeg, events, 
        tmin=0,     # init timestamp of epoch (0 means trigger timestamp same as event start)
        tmax=10,    # final timestamp (10 means set epoch duration 10 second)
        baseline=(0, 0),
        preload=True,
    )

    # Visualization
    if show_eeg:
        raw_eeg.plot(
                duration=10, 
                start=0, 
                scalings=plot_scale, # You may edit scalings value later
            ) #, n_channels=8, bad_color='red'
            # show = True

        epochs['2'].plot(
            scalings=plot_scale, # You may edit scalings value later
            title='Left stimuli start',
        )
        epochs['5'].plot(
            scalings=plot_scale, # You may edit scalings value later
            title='Right stimuli start',
        )

    fig, ax = plt.subplots(2, figsize=(9,9))

    epochs['2'].compute_psd(
        fmax=30,                    
        ).plot(
            axes=ax[0],
            average=True,
            )
    ax[0].set_title('Left stimuli' if not filename else 'Left stimuli - '+filename)

    epochs['5'].compute_psd(
        fmax=30,                    
        ).plot(
            axes=ax[1],
            average=True,
            )
    ax[1].set_title('Right stimuli' if not filename else 'Right stimuli - '+filename)

    plt.tight_layout()
    plt.show()


if __name__=='__main__':

    # raw = read_xdf("example.xdf")
    # raw = read_xdf("test01_OpenBCI.xdf", show_plot=True, show_psd=True)

    filename = 'Pipo_1_5_test1.xdf'
    # filename = 'Pipo_1_5_test2.xdf'
    # filename = 'Pipo_1_5_test3.xdf'

    raw = read_xdf(filename, 
            show_plot=False, 
        show_psd=False,
    )

    show_epoch(raw, filename)