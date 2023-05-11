import pyxdf
from pyxdf import match_streaminfos, resolve_streams

import mne
from mnelab.io.xdf import read_raw_xdf

import matplotlib.pyplot as plt
from pprint import pprint


def read_xdf(filename: str, bandpass=(None, 45.0), show_plot=True, show_psd=True, verbose=False, plot_scale=169) -> mne.io.array.array.RawArray:
    """
    Loading XDF file into MNE-RawArray. MNE-Python does not support this file format out of the box, 
    but we can use the pyxdf package and MNELAB to import the data. 

    attribute:
        bandpass  : set Bandpass filter (l_freq, h_freq)
        show_plot : If True, show all EEG channels and able to zoom in-out, scaling
        show_psd  : If True, show overall average power spectral density

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

    # Add Bandpass filtering (default 0Hz - 45Hz)
    raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

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


def show_epoch(raw: mne.io.array.array.RawArray, filename=None, show_eeg=False, show_time_freq=False, plot_scale=200):
    """
    Showing Power spectral density (PSD) plot, split by Left-Right stimuli event, average by epoch 

    attribute:
        show_eeg        : If True, (same as show_plot) show all EEG channels and able to zoom in-out, scaling
        show_time_freq  : If True, show Time-Frequency plot split by Left-Right stimuli and each O1, Oz, O2, POz, Pz

    return: None
    """
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
        # method='welch',
        ).plot(
            axes=ax[0],
            average=True, 
            )
    ax[0].set_title('Left stimuli' if not filename else 'Left stimuli - '+filename)

    epochs['5'].compute_psd(
        fmax=30,                    
        # method='welch',
        ).plot(
            axes=ax[1],
            average=True, 
            )
    ax[1].set_title('Right stimuli' if not filename else 'Right stimuli - '+filename)
    plt.tight_layout()

    # Plot Time-frequency
    if show_time_freq:
        # Split Epochs (Trials) including Cue
        epochs_from_cue = mne.Epochs(raw_eeg, events, 
                tmin=-1.0,      # init timestamp of epoch (-1.0 means trigger before event start 1.0 second)
                tmax=10.0,      # final timestamp (10 means set epoch duration 10 second)
                # baseline=(0, 0),
                preload=True,
            )
        channel_name = (    
            '(O1) obci_eeg1_1',
            '(Oz) obci_eeg1_2',
            '(O2) obci_eeg1_3',
            '(POz) obci_eeg1_4',
            '(Pz) obci_eeg1_5',)

        fig, ax = plt.subplots(2, 5, figsize=(17, 7))
        plt.title('Time-frequency')

        # Select range of frequency you wanna plot
        show_freqs = list(range(1,25))
        # show_freqs = [6.0, 12.0, 18.0, 24.0, 10.0, 20.0, 30.0]

        for i in range(5):
            power_L = mne.time_frequency.tfr_multitaper(
                epochs_from_cue['2'], 
                freqs=show_freqs, 
                n_cycles=10, 
                use_fft=True, 
                decim=3,
                return_itc=False,
            )
            power_L.plot([i], mode='logratio', axes=ax[0,i], show=False, colorbar=False)
            ax[0,i].set_title('Left stimuli - '+channel_name[i]); ax[0,i].set_xlabel('')

            power_R = mne.time_frequency.tfr_multitaper(
                epochs_from_cue['5'], 
                freqs=show_freqs, 
                n_cycles=10, 
                use_fft=True, 
                decim=3,
                return_itc=False,
            )
            power_R.plot([i], mode='logratio', axes=ax[1,i], show=False, colorbar=False)
            ax[1,i].set_title('Right stimuli - '+channel_name[i])
        plt.tight_layout()

    plt.show()


if __name__=='__main__':

    filename = 'Pipo_1_5_test1.xdf'
    # filename = 'Pipo_1_5_test2.xdf'
    # filename = 'Pipo_1_5_test3.xdf'

    # Loading XDF file into MNE-RawArray
    raw = read_xdf(filename, 
        bandpass=(3.0, 15.0), # (default 0Hz - 45Hz)

        show_plot=True, 
        # show_plot : If True, show all EEG channels and able to zoom in-out, scaling

        show_psd=False,
        # show_psd : If True, show overall average power spectral density
    )

    # Showing Power spectral density (PSD) split by Left-Right stimuli event
    show_epoch(raw, filename,

        show_eeg=False,
        # show_eeg : If True, show all EEG channels and able to zoom in-out, scaling split by Left-Right stimuli

        show_time_freq=True
        # show_time_freq : If True, show Time-Frequency plot split by Left-Right stimuli and each O1, Oz, O2, POz, Pz
    )