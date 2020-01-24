[![Build Status](https://travis-ci.com/AznStevy/Heart-Rate-Monitor.svg?branch=master)](https://travis-ci.com/AznStevy/Heart-Rate-Monitor)

# Heart Rate Monitor
Heart rate monitors have been an essential tool in evaluating the health of a person as well as an integral part of diagnosis for cardiac disorders ranging from arrhythmia to conditions like Wellen's syndrome.

This is code that acts as a heart rate analyzer (rather than a monitor). It can determine heartbeats based on two different algorithms: Thresholding and Wavelet transform + Thresholding. I've found the Wavelet method to work the best. The goal here was for anyone to write their own detection method and be able to use it with the `HeartRateMonitor` class.

## Installation
Using a virtual environment with a python environmental variable, run the command below to get the necessary packages: ```pip install -r requirements.txt```

## Structure
### `FileHandler`
This class is a file handler specifically used for a specific kind of csv which contains two strips of data, one being a time array and the other being the signal. Upon initialization, the class checks the dimensions and orientation of the data to ensure it can be properly processed. It then checks for missing and/or non-numeric values and gets rid of them. Lastly, it separates the data int to `time` and `raw_signal` attributes to be used by other classes.

### `detection_algorithm`
#### ECGDetectionAlgorithm
The `ECGDetectionAlgorithm` is meant to be somewhat abstract class which is only differentiated by its `find_mean_hr_bpm`, `find_beats`, and `plot_graph` methods. The core functionality and attributes of the final return which can be easily handled are handled here such as `find_voltage_extremes`, `find_duration`, and `find_num_beats` (which requires the `find_beats` abstract method, but it must be implemented in extended classes).

#### Threshold
The `Threshold` class is one of the two methods of detection. It takes the raw signal, cleans it using `apply_noise_reduction` via background subtraction (moving average) as well as band-pass filtering based on literature. The base threshold is set to the maximum of the filtered signal by default and modulated based on the signal itself. The overlap between the threshold and filtered signal is then converted into a binary array using `apply_threshold` where `1` denotes overlap. The helper method `_find_binary_centers` locates the rising and falling edges of the binary signal and then calculates the central index between the two, giving a single spike.

#### Wavelet
The `Wavelet` class extends `Threshold` and essentially uses the same thresholding method. The only difference here is the signal that is passed through the thresholding algorithm. As opposed to using a filtered signal, the class uses a wavelet transform of a minimal range of widths. The 1-dimensional average tends to provide a more obvious peak for thresholding, yielding slightly better results.

### `HeartRateMonitor`
The `HeartRateMonitor` class is meant to be the outer-most class to be instantiated by a user. In order to get the necessary information

## Usage
Using the `HeartRateMonitor` class, you input the signal filename as well as the detection method. To get an output file, call the `write_json()` method. To get the `metrics` with the `beats` in `numpy.ndarray` form, call the `to_json()` method.

### Example
```python
from detection_algorithm import Wavelet
from heart_rate_monitor import HeartRateMonitor

def main():
    # if you wanted to specify a width for mean hr,
    # use the time_interval kwarg. e.g. time_interval=(.13, .16)
    # The Wavelet analyzer can be swapped with Threshold
    heart_rate_monitor = HeartRateMonitor(
        filename="tests/test_data/test_data1.csv",
        analyzer=Wavelet)

    if heart_rate_monitor is None:
        return

    # if you do not wish to write the heart rate properties, use to_json()
    metrics = heart_rate_monitor.to_json()
    heart_rate_monitor.write_json()

if __name__ == "__main__":
    main()
```


## Plotting
If you wish to get a plot of the analysis, use the `plot_graph` method on the analyzer e.g. `heart_rate_monitor.analyzer.plot_graph()`.

### Example
Using `heart_rate_monitor.analyzer.plot_graph()` on a Wavelet analyzer would give you a plot similar to the one below.
![Example Figure](https://i.imgur.com/vvmEqRl.png)

