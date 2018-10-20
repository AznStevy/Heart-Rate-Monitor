from file_handler import FileHandler
from detection_algorithm import Threshold, Convolution, Wavelet


class HeartRateMonitor(object):
    def __init__(self, filename, analyzer):
        filename = filename
        self.file_handler = FileHandler(filename)
        self.time = self.file_handler.time
        self.raw_signal = self.file_handler.signal
        self.analyzer = analyzer(self.time, self.raw_signal,
                                 name='{}'.format(self.file_handler.filename))
        self.analyzer.start_analysis()

    def to_json(self):
        """
        Converts relevant properties into a json/dict object.
        Returns: json object of relevant properties.

        """
        dict_obj = {
            "beats": self.analyzer.beats,
            "duration": self.analyzer.duration,
            "num_beats": self.analyzer.num_beats,
            "mean_hr_bpm": self.analyzer.mean_hr_bpm,
            "voltage_extremes": self.analyzer.voltage_extremes
        }
        return dict_obj

    def write_json(self):
        """
        Writes json to a file called the same base name with .json extension.
        """
        metrics = self.to_json()
        filename = "{}.json".format(self.file_handler.basename)
        self.file_handler.write_data(metrics, filename)


if __name__ == "__main__":
    # 8, 9 (Wellens disease/inverse signal), 12, 15, 24 (weird signal),
    for i in range(1):
        num = i + 1
        heart_rate_monitor = HeartRateMonitor(
            filename="tests/test_data/test_data{}.csv".format(num),
            analyzer=Threshold)

        metrics = heart_rate_monitor.to_json()
        heart_rate_monitor.analyzer.plot_graph()
        heart_rate_monitor.write_json()
        print(num, metrics["mean_hr_bpm"], metrics["num_beats"])
