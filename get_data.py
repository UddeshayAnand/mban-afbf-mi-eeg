"""Notebook loaders; support flat and Training/Evaluation dataset folders."""
import os
from pathlib import Path
import numpy as np
import scipy.io as sio

def resolve_file(file_path, training):
    path = Path(file_path)
    if not path.is_file():
        path = path.parent / ("Training" if training else "Evaluation") / path.name
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return path

def load_BCI2a_data(data_path, subject, training, all_trials=True):
    n_channels = 22
    n_tests = 6 * 48
    window_Length = 7 * 250

    fs = 250
    t1 = int(2 * fs)
    t2 = int(6 * fs)

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    suffix = "T" if training else "E"
    file_path = os.path.join(data_path, f"A0{subject}{suffix}.mat")

    a = sio.loadmat(resolve_file(file_path, training))
    a_data = a['data']

    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data3 = a_data1[0, 0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if a_artifacts[trial] != 0 and not all_trials:
                continue
            data_return[NO_valid_trial, :, :] = np.transpose(a_X[int(np.asarray(a_trial[trial]).item()):(int(np.asarray(a_trial[trial]).item()) + window_Length), :22])
            class_return[NO_valid_trial] = int(np.asarray(a_y[trial]).item())
            NO_valid_trial += 1

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)

    return data_return, class_return

def load_BCI2b_data(data_path, subject, training):
    """
    Load one subject's data from BCI Competition IV Dataset 2b.
    Only artifact-free trials are kept (a_artifacts == 0).

    FIX: Uses dynamic list accumulation instead of a pre-allocated array.
    This prevents IndexError for subjects (e.g. 4, 8) whose .mat files
    contain more sessions/runs than a fixed-size array can accommodate.

    t1:t2  →  2 s – 6 s  =  1000 samples @ 250 Hz
    """
    n_channels    = 3
    window_length = 8 * 250   # full window size in the .mat file
    t1 = int(3 * 250)         # 500  — crop start
    t2 = int(7 * 250)         # 1500 — crop end  →  1000-sample segment

    suffix    = "T" if training else "E"
    file_path = os.path.join(data_path, f"B0{subject}{suffix}.mat")

    a      = sio.loadmat(resolve_file(file_path, training))
    a_data = a["data"]

    # -----------------------------------------------------------------
    # FIX: accumulate into lists — no pre-allocation, no fixed upper bound
    # -----------------------------------------------------------------
    data_list  = []
    class_list = []

    for ii in range(a_data.size):
        a_data1     = a_data[0, ii]
        a_data3     = a_data1[0, 0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(a_trial.size):
            artifact = int(np.array(a_artifacts[trial]).item())
            if artifact == 0:                                     # only valid trials
                start   = int(np.array(a_trial[trial]).item())
                segment = np.transpose(
                    a_X[start:(start + window_length), :n_channels]
                )                                                 # (n_channels, window_length)
                data_list.append(segment[:, t1:t2])              # crop to 1000 samples
                class_list.append(int(np.array(a_y[trial]).item()))

    data_return  = np.array(data_list)                           # (n_valid, 3, 1000)
    class_return = (np.array(class_list) - 1).astype(int)       # 1-based → 0-based

    print(f"  Subject {subject} {'Train' if training else 'Test'}: "
          f"{len(data_list)} artifact-free trials loaded")
    return data_return, class_return

def get_data(path, subject, dataset="2a"):
    if subject not in range(1, 10):
        raise ValueError("Subject must be between 1 and 9")
    loader = load_BCI2a_data if dataset == "2a" else load_BCI2b_data
    x_train, y_train = loader(path, subject, True)
    x_test, y_test = loader(path, subject, False)
    return x_train, y_train, x_test, y_test
