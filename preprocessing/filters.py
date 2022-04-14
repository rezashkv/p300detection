from scipy.signal import firwin
import numpy as np


def reduce_noise(data, fs, fc1, fc2):
    num_channels = data.shape[0]
    num_taps = 6 * fs
    num_taps |= 1
    fir_filter = firwin(num_taps, [fc1 / (fs / 2), fc2 / (fs / 2)])
    outputs = np.zeros(data.shape)
    for i in range(num_channels):
        outputs[i, :] = np.convolve(data[i, :], fir_filter, mode='same')
    return outputs

