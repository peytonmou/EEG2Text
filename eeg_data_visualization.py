import mne
from mne.time_frequency import tfr_multitaper, psd_array_multitaper
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}
EEG_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
PLOT_STYLE = {'style': 'whitegrid', 'context': 'talk', 'palette': 'tab10'}

def setup_environment():
    """Initialize the environment and load data"""
    filefolder = 'your_filepath' 
    return [f for f in os.listdir(filefolder) if f.endswith('.edf')]

def load_and_preprocess(edf_file, filefolder):
    """Load and preprocess EDF file"""
    label = edf_file.split('_')[-1].split('.')[0].capitalize()
    filepath = os.path.join(filefolder, edf_file)
    raw = mne.io.read_raw_edf(filepath, preload=True)
    
    # Set channel types and montage
    for ch in raw.info['ch_names']:
        raw.set_channel_types({ch: 'eeg' if ch in EEG_CHANNELS else 'misc'})
    raw.set_montage('standard_1020')
    raw.filter(1, 40, fir_design='firwin')
    return raw, label

def create_epochs(raw, duration=10):
    """Create epochs from raw data"""
    try:
        events = mne.make_fixed_length_events(raw, duration=duration)
        return mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True) if len(events) > 0 else None
    except ValueError as e:
        print(f"Error creating events: {e}")
        return None

def analyze_power(raw, label, power_data):
    """Analyze power spectrum and collect data for topomaps"""
    eeg_data = raw.get_data(picks='eeg')
    sfreq = raw.info['sfreq']
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        psds, _ = psd_array_multitaper(eeg_data, sfreq=sfreq, fmin=fmin, fmax=fmax)
        power_data[label][band_name].append(psds.mean(axis=1))

def plot_topomaps(power_data, info, title=""):
    """Plot topomaps for power data across frequency bands"""
    n_rows, n_cols = len(FREQ_BANDS), len(power_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    
    for col_idx, label in enumerate(sorted(power_data.keys())):
        for row_idx, (band_name, psd_avg) in enumerate(power_data[label].items()):
            mne.viz.plot_topomap(psd_avg, info, cmap='viridis', size=2, 
                               axes=axes[row_idx, col_idx], show=False)
            axes[row_idx, col_idx].set_title(f'{band_name} ({label})', fontsize=10)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_erp(erp_data, channel='T8', title="Time-Domain Analysis"):
    """Plot ERP waveforms for all labels"""
    sns.set(**PLOT_STYLE)
    colors = sns.color_palette(PLOT_STYLE['palette'], len(erp_data))
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, (label, evoked) in enumerate(erp_data.items()):
        ax = axes[idx]
        times, data = evoked.times, evoked.get_data(picks=channel)[0]
        ax.plot(times, data, label=label, color=colors[idx], linewidth=2)
        ax.set_title(f'Label: {label}', fontsize=14)
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
    
    fig.suptitle(f'{title} (Channel: {channel})', fontsize=18)
    fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Amplitude (µV)', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def analyze_time_frequency(epochs):
    """Perform time-frequency analysis"""
    freqs = np.arange(1, 41, 1)
    return tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs/2, return_itc=False, average=False)

def plot_time_frequency(tfr_data, channel='O2', title="Time-Frequency Analysis"):
    """Plot time-frequency representations"""
    channel_idx = EEG_CHANNELS.index(channel)
    plt.figure(figsize=(15, 10))
    
    for idx, (label, tfr_avg) in enumerate(tfr_data.items()):
        if tfr_avg.shape[0] <= channel_idx:
            continue
            
        plt.subplot(2, 5, idx + 1)
        plt.imshow(10 * np.log10(tfr_avg[channel_idx]),
                   extent=[0, 10, 1, 40], aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'{label} ({channel})', fontsize=12)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Frequency (Hz)', fontsize=10)
        plt.colorbar(label='Power (dB)')
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    edf_files = setup_environment()
    filefolder = '/content/drive/MyDrive/Char'
    
    # Initialize data structures
    power_data = {label: {band: [] for band in FREQ_BANDS} 
                 for label in set([f.split('_')[-1].split('.')[0].capitalize() for f in edf_files])}
    erp_data = {}
    tfr_data = {}
    
    # Process each file
    for edf_file in edf_files:
        raw, label = load_and_preprocess(edf_file, filefolder)
        
        # Power analysis for topomaps
        analyze_power(raw, label, power_data)
        
        # ERP analysis
        epochs = create_epochs(raw)
        if epochs and len(epochs) > 0:
            if label not in erp_data:
                erp_data[label] = []
            erp_data[label].append(epochs)
            
            # Time-frequency analysis
            if label not in tfr_data:
                tfr_data[label] = []
            tfr_data[label].append(analyze_time_frequency(epochs))
    
    # Average and plot results
    averaged_power = {
        label: {band: np.mean(np.stack(power_list), axis=0) 
               for band, power_list in band_data.items()}
        for label, band_data in power_data.items()
    }
    plot_topomaps(averaged_power, raw.info, "Average Power Topomaps")
    
    averaged_erp = {
        label: mne.concatenate_epochs(epochs_list).average()
        for label, epochs_list in erp_data.items()
    }
    plot_erp(averaged_erp)
    
    averaged_tfr = {
        label: np.mean(np.stack([tfr.data for tfr in tfr_list]), axis=0)
        for label, tfr_list in tfr_data.items()
    }
    plot_time_frequency(averaged_tfr)

if __name__ == "__main__":
    main() 