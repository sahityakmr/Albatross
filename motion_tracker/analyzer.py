import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter


def get_rolling_mean(x):
    avg = np.mean(x)
    rolling_avg = x.rolling(10).mean()
    rolling_avg = [avg if np.isnan(x) else x for x in rolling_avg]
    rolling_avg = [1.2 * x for x in rolling_avg]
    return rolling_avg


# rm_x = get_rolling_mean(dataset['x'])
# rm_y = get_rolling_mean(dataset['y'])
# rm_z = get_rolling_mean(dataset['z'])


def low_pass_filter(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def filter_signal(x):
    b, a = low_pass_filter(2, 40, order=10)
    return lfilter(b, a, x)


ACCELEROMETER_FILE = 'res/accelerometer/accelerometer.csv'
MAGNETOMETER_FILE = 'res/magnetometer/magnetometer.csv'
GYROSCOPE_FILE = 'res/gyroscope/gyroscope.csv'

MAGNETOMETER = 'magnetometer'
ACCELEROMETER = 'accelerometer'
GYROSCOPE = 'gyroscope'

dataset_dictionary = {
    MAGNETOMETER: pd.read_csv(MAGNETOMETER_FILE),
    ACCELEROMETER: pd.read_csv(ACCELEROMETER_FILE),
    GYROSCOPE: pd.read_csv(GYROSCOPE_FILE)
}

for key in dataset_dictionary:
    dataset = dataset_dictionary[key]
    plt.plot(dataset['x'], color='black')
    plt.plot(dataset['y'], color='blue')
    plt.plot(dataset['z'], color='green')
    plt.plot(filter_signal(dataset['x']), color='yellow')
    plt.plot(filter_signal(dataset['y']), color='orange')
    plt.plot(filter_signal(dataset['z']), color='pink')
    plt.show()
