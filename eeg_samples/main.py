import json
import numpy as np
import matplotlib.pyplot as plt

from pprint import PrettyPrinter
pp = PrettyPrinter()

import mne
from mne.io import RawArray

# Files are 
# 1a. 1_D.txt
# 2a. 1_D_converted.txt
# 1b. 1_T.txt
# 2b. 1_T_converted.txt
#
# The "converted files" appear to be the same as the original
# files, except they appear to be normalized in some way.
SAMPLE_DANIELA_FNAME = "1_D_converted.txt"
SAMPLE_TANZEELA_FNAME = "1_T_converted.txt"

# Literally returns the lines of a file in a generator
def file_lines(filename):
    with open(filename, "r") as f:
        for line in f.readlines():
            yield line

def line2_info(filename):
    with open(filename, "r") as f:
        return json.loads(f.readlines()[1][1:].strip())

# Removes the first 3 lines of the files that are comments
def filter_comments(lines):
    return filter(lambda line: line[0] != "#", lines)

# Removes the 4-bit counter as well as the first 4 unused EEG measurements
def remove_unused_datapoints(lines):
    return map(lambda line: line.strip().split()[-1].strip(), lines)

# Convert a timeseries of strings representing floats into actual floats
def convert_str_ts_to_float(lines):
    return map(lambda strfl: float(strfl), lines)

# This will extract a timeseries of raw (normalized) values from a timeseries
# and return it as a list (for now)
def extract_ts(filename):
    # We expect (number_channels, number_times)
    # https://mne.tools/stable/generated/mne.io.RawArray.html
    return np.asarray(list(
        convert_str_ts_to_float(
            remove_unused_datapoints(
                filter_comments(
                    file_lines(filename)))))).reshape((1, -1))

DEBUG_PLOT_RAW = False
if __name__ == "__main__":
    # These are normalized raw signals (I think)
    _daniela_ts = extract_ts(SAMPLE_DANIELA_FNAME)
    _tanzeela_ts = extract_ts(SAMPLE_TANZEELA_FNAME)

    # Get their info (from the measurement)
    _daniela_info = line2_info(SAMPLE_DANIELA_FNAME)
    _tanzeela_info = line2_info(SAMPLE_TANZEELA_FNAME)

    # You'll see that we have
    # 1 channel
    # 1000 as our sampling rate
    # time (around 1-1:25)
    # and some other things...
    print("Daniela ts shape: {}".format(_daniela_ts.shape))
    pp.pprint(_daniela_info)
    print("Tanzeela ts shape: {}".format(_tanzeela_ts.shape))
    pp.pprint(_tanzeela_info)

    # You will find that these are very noisy. We will want to filter them
    # somehow using a band pass.
    if DEBUG_PLOT_RAW:
        plt.plot(_daniela_ts)
        plt.show()
        plt.clf()
        plt.plot(_tanzeela_ts)
        plt.show()
        plt.clf()
    
    # TODO we are trying to break this up into alpha, beta, gamma, delta, and theta waves
    bitalino_name = '98:D3:91:FD:3F:F3'
    # https://mne.tools/stable/generated/mne.create_info.html#mne.create_info
    # Look at the pprint above to see what the different mentadata we have available
    # to provide are. We use "label" instead of "channels" for the string.
    daniela_mne_info = mne.create_info(_daniela_info[bitalino_name]['label'],
        float(_daniela_info[bitalino_name]['sampling rate']),ch_types=['eeg'])
    tanzeela_mne_info = mne.create_info(_tanzeela_info[bitalino_name]['label'],
        float(_tanzeela_info[bitalino_name]['sampling rate']), ch_types=['eeg'])

    # https://mne.tools/stable/generated/mne.io.RawArray.html
    daniela = RawArray(_daniela_ts, daniela_mne_info)
    tanzeela = RawArray(_tanzeela_ts, tanzeela_mne_info)
    print(daniela.info)
    print(tanzeela.info)

    # This is very stupid :)
    # NOTE we still need to cut off the noise artifacts...
    idx2name = {0:"alpha", 1:"beta", 2:"gamma", 3:"delta", 4:"theta"}
    cutoffs = [(7, 13)]#, (13, 39), (40, None), (None, 4), (4, 7)]
    daniela_waves = [daniela.filter(lo, hi, fir_design='firwin', skip_by_annotation='edge') for lo, hi in cutoffs]
    # tanzeela_waves = [tanzeela.filter(lo, hi, fir_design='firwin', skip_by_annotation='edge') for lo, hi in cutoffs]

    # TODO for some reason these are all the same
    daniela_alpha = daniela_waves[0]
    # daniela_beta = daniela_waves[1]
    print(daniela[:])
    print("\n")
    print(daniela_alpha[:])
    # print(daniela_beta.shape)
    # TODO Muntaser claims that he's doing this

    # WTF
    # for wave in daniela_waves:
    #     wave.plot(duration=5, n_channels=1)

    # daniela.plot_psd(fmax=50)
    # daniela.plot(duration=5, n_channels=1)