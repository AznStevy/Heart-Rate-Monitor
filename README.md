# bme590hrm
[![Build Status](https://travis-ci.com/AznStevy/bme590hrm.svg?branch=szx2%2Fdevelop)](https://travis-ci.com/AznStevy/bme590hrm)

## Heart Rate Monitor
Heart rate monitors have been an essential tool in evaluating the health of a person as well as an integral part of diagnosis for cardiac disorders ranging from arrhythmia to conditions like Wellen's syndrome.

This is code that acts as a heart rate analyzer (rather than a monitor). It can determine heartbeats based on three different algorithms: Thresholding, Convolution + Thresholding, and Wavelet transforms + Thresholding. I've found the Wavelet method to work the best.

## Usage
Using the `HeartRateMonitor` class, you input the signal filename as well as the detection method. To get an output file, call the `write_json()` method.

### Example
```python
from detection_algorithm import Wavelet
from heart_rate_monitor import HeartRateMonitor

def main():
    heart_rate_monitor = HeartRateMonitor(
        filename="tests/test_data/test_data1.csv",
        analyzer=Wavelet)

    if heart_rate_monitor is None:
        continue

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

