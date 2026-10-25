"""Preserve the notebook segmentation, pooled split, and dataset-specific filters."""
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from get_data import get_data

def sliding_window(data, sfreq=250, overlap=0.95, shift_sec=0.1):
    """
    Slice EEG trials into overlapping windows.
    Parameters
    ----------
    data : ndarray, shape (n_trials, n_chans, n_times)
    sfreq : float
        Sampling frequency in Hz.
    overlap : float
        Fractional overlap between windows.
    shift_sec : float
        Step size between window starts in seconds.
    Returns
    -------
    out : ndarray, shape (n_trials*n_windows, n_chans, window_length)
    """
    print(data.shape)
    n_trials, n_chans, n_times = data.shape
    shift = int(round(shift_sec * sfreq))                # e.g. 0.1s -> 25 samples
    window_length = int(round(shift / (1 - overlap)))    # e.g. 500 samples for 95% overlap
    n_windows = (n_times - window_length) // shift + 1    # number of windows per trial

    # Preallocate output array
    out = np.zeros((n_trials * n_windows, n_chans, window_length), dtype=data.dtype)
    idx = 0
    for tr in range(n_trials):
        for w in range(n_windows):
            start = w * shift
            stop = start + window_length
            out[idx] = data[tr, :, start:stop]
            idx += 1
    return out

def apply_band_filters(X_data, fs=250, order=4):
    """
    Applies bandpass filtering to EEG data shaped (samples, timepoints, channels).
    Returns a tuple of filtered signals in different bands.
    """
    # 1. Remove DC offset across time axis (axis=1)
    X_data = X_data - np.mean(X_data, axis=1, keepdims=True)

    # 2. Band definitions
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    # 3. Reshape to [samples * channels, timepoints] for filtering along time
    n_samples, n_time, n_channels = X_data.shape
    X_reshaped = X_data.transpose(0, 2, 1).reshape(-1, n_time)

    band_data = {}
    for band_name, (low, high) in bands.items():
        nyq = 0.5 * fs
        Wn = [low / nyq, high / nyq]

        if Wn[1] >= 1.0:
            raise ValueError(f"Invalid high cutoff {high}Hz for {fs}Hz sampling")

        b, a = butter(order, Wn, btype='bandpass')

        padlen = min(150, X_reshaped.shape[1] // 2)
        filtered = filtfilt(b, a, X_reshaped, axis=1, padlen=padlen)

        # Reshape back to original: (samples, time, channels)
        band_filtered = filtered.reshape(n_samples, n_time, n_channels).transpose(0, 2, 1)
        band_data[band_name] = band_filtered

    return (
        band_data['delta'],
        band_data['theta'],
        band_data['alpha'],
        band_data['beta'],
        band_data['gamma']
    )

def apply_band_filters_2b(X_data, fs=250, order=4):
    """
    Input : (samples, timepoints, channels)
    Output: delta, theta, alpha, beta, gamma  —  each (samples, timepoints, channels)
    """
    X_data = X_data - np.mean(X_data, axis=1, keepdims=True)   # remove DC offset

    bands = {
        'delta': (0.5,  4),
        'theta': (4,    8),
        'alpha': (8,   13),
        'beta':  (13,  30),
        'gamma': (30, 100),
    }

    n_samples, n_time, n_channels = X_data.shape
    X_reshaped = X_data.transpose(0, 2, 1).reshape(-1, n_time)  # (N*C, T)

    band_data = {}
    for band_name, (low, high) in bands.items():
        nyq  = 0.5 * fs
        Wn   = [low / nyq, high / nyq]
        if Wn[1] >= 1.0:
            raise ValueError(f"Invalid high cutoff {high} Hz for {fs} Hz sampling rate")
        b, a     = butter(order, Wn, btype='bandpass')
        padlen   = min(150, X_reshaped.shape[1] // 2)
        filtered = filtfilt(b, a, X_reshaped, axis=1, padlen=padlen)
        band_data[band_name] = (
            filtered.reshape(n_samples, n_channels, n_time)
                    .transpose(0, 2, 1)         # → (samples, time, channels)
        )

    return (
        band_data['delta'],
        band_data['theta'],
        band_data['alpha'],
        band_data['beta'],
        band_data['gamma'],
    )

def prepare_data(path, subject, dataset="2a"):
    xtr, ytr, xte, yte = get_data(path, subject, dataset)
    tr, te = sliding_window(xtr), sliding_window(xte)
    x = np.concatenate((tr, te)).transpose(0, 2, 1)
    y = np.concatenate((np.repeat(ytr, len(tr)//len(xtr)),
                        np.repeat(yte, len(te)//len(xte))))
    x_tv, x_test, y_tv, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tv, y_tv, test_size=0.1/0.8, stratify=y_tv, random_state=42)
    splits = {}
    for name, data, labels in [("train", x_train, y_train),
                               ("val", x_val, y_val), ("test", x_test, y_test)]:
        # 2a intentionally retains the notebook's channel-axis filtering/reshape.
        bands = (apply_band_filters(data.transpose(0, 2, 1)) if dataset == "2a"
                 else apply_band_filters_2b(data))
        splits[name] = ({"input_main": data, "input_theta": bands[1],
                         "input_alpha": bands[2], "input_beta": bands[3]}, labels)
    return splits

def model_inputs(model, data):
    return [data[t.name.split(":")[0]] for t in model.inputs]
