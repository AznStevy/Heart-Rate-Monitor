import os
import logging
from file_handler import FileHandler
from detection_algorithm import ECGDetectionAlgorithm, Threshold, Wavelet

logging.basicConfig(filename='heart_rate_monitor.log', level=logging.DEBUG)


class HeartRateMonitor(object):
    def __init__(self, filename, analyzer, **kwargs):
        logging.info("Heart rate monitor instantiated.")
        filename = filename

        self.file_handler = FileHandler(filename)
        self.time = self.file_handler.time
        self.raw_signal = self.file_handler.signal

        if isinstance(analyzer, ECGDetectionAlgorithm):
            raise TypeError("Analyzer must be type ECGDetectionAlgorithm.")

        self.analyzer = analyzer(self.time, self.raw_signal,
                                 name='{}'.format(self.file_handler.filename))

        time_interval = kwargs.get('time_interval', None)
        self.analyzer.start_analysis(time_interval=time_interval)

    def to_json(self) -> dict:
        """
        Converts relevant properties into a json/dict object.
        Returns:
            dict: json object of relevant properties.
        """
        logging.info("to_json called")
        dict_obj = {
            "beats": self.analyzer.beats,
            "duration": self.analyzer.duration,
            "num_beats": self.analyzer.num_beats,
            "mean_hr_bpm": self.analyzer.mean_hr_bpm,
            "voltage_extremes": self.analyzer.voltage_extremes
        }
        return dict_obj

    def write_json(self) -> str:
        """
        Writes json to a file called the same base name with .json extension.

        Returns:
            str: The output filename.
        """
        logging.info("write_json called")
        metrics = self.to_json()
        filename = "{}.json".format(self.file_handler.basename)
        try:
            self.file_handler.write_data(metrics, filename)
        except TypeError as e:
            logging.exception(e)
        logging.info("Wrote json to {}".format(filename))
        return filename


def main():
    # 8, 9 (Wellens disease/inverse signal), 12, 15, 16, 24 (weird signal), 29
    for i in range(31):
        num = i + 1
        try:
            heart_rate_monitor = HeartRateMonitor(
                filename="tests/test_data/test_data{}.csv".format(num),
                analyzer=Wavelet)

            if heart_rate_monitor is None:
                continue
        except TypeError as e:
            logging.exception(e)
            continue

        metrics = heart_rate_monitor.to_json()
        heart_rate_monitor.analyzer.plot_graph()
        heart_rate_monitor.write_json()
        print(num, metrics)


if __name__ == "__main__":
    main()
