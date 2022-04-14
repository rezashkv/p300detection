import numpy as np
from scipy.io import loadmat
from p300detection.preprocessing.filters import reduce_noise

dataset = loadmat('../../Project/preprocessing/Subject_A_Train.mat')
raw_signal = np.transpose(dataset['Signal'], (0, 2, 1))
stimulus_type = dataset['StimulusType'].astype(int)
stimulus_code = dataset['StimulusCode'].astype(int)

SAMPLING_FREQUENCY = 240

TIME_AFTER_STIMULI = 800  # MILLISECONDS
WINDOW_LENGTH = TIME_AFTER_STIMULI * SAMPLING_FREQUENCY // 1000  # NUMBER OF EPOCHS SAMPLES

TIME_BEFORE_STIMULI = 100  # MILLISECONDS
WINDOW_LENGTH_BASELINE = TIME_BEFORE_STIMULI * SAMPLING_FREQUENCY // 1000  # NUMBER OF BASELINE SAMPLES

n_channels = raw_signal.shape[1]
n_chars = 85
n_trials = 180

labels = np.zeros(n_chars * n_trials)
data = np.zeros((n_chars * n_trials, n_channels, WINDOW_LENGTH))

event = []
i = 0
for char in range(n_chars):
    signal = np.squeeze(raw_signal[char, :, :])

    FC1, FC2 = 0.5, 40
    filtered_signal = reduce_noise(signal, SAMPLING_FREQUENCY, FC1, FC2)
    target_t = np.argwhere(stimulus_type[char, :] == 1).squeeze()
    step_size = 24
    target = target_t[np.arange(0, len(target_t), step_size)]

    stimulus_c = np.argwhere(stimulus_code[char, :] != 0).squeeze()
    stimulus = stimulus_c[np.arange(0, len(stimulus_c), step_size)]
    event.append(stimulus_code[char, stimulus])
    q = np.array([np.argwhere(stimulus == i)[0, 0] for i in target])
    labels[q + i * stimulus.shape[0]] = 1

    for j in range(stimulus.shape[0]):
        if stimulus[j] == 1:
            WINDOW_LENGTH_BASELINE = 0
        data[j + i * len(stimulus), :, :] = filtered_signal[:, stimulus[j]: stimulus[j] + WINDOW_LENGTH]
        if j > 0:
            data[j + i * len(stimulus), :, :] -= np.mean(filtered_signal[:, stimulus[j] - WINDOW_LENGTH:stimulus[j]], 1).reshape(64, 1)
    i += 1
