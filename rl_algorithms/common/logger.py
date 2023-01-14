import sys
from pathlib import Path
from typing import List, Union
import csv

import datetime

import pandas as pd


class Logger:
    """Provides a logger class for logging the results."""
    
    # TODO: add algorithm name to data

    def __init__(self, log_dir=None, file_name=None) -> None:
        """Inits the Logger."""
        self.log_dir = log_dir
        self.file_name = file_name

        date = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")

        if not self.log_dir:
            self.log_dir = Path(sys.modules["__main__"].__file__).parent / "logs"

        if not self.file_name:
            self.file_name = f"{date}.csv"

        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.log_filepath = self.log_dir / self.file_name
        if self.log_filepath.exists():
            new_log_filepath = self.log_filepath.replace(".csv", f"_{date}.csv")
            print(
                f"Logfile {self.log_filepath} already exists."
                f"Logfile will be saved to {new_log_filepath}."
            )
            self.log_filepath = new_log_filepath

        self.log_filepath.touch()

    def write(self, data: List[Union[str, int, float]]) -> None:
        """Writes to log file."""
        with open(self.log_filepath, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)



def get_data(logdir: Path) -> List[pd.DataFrame]:
    """Gets the data from all log files in the directory."""
    data = []
    filepaths = logdir.glob(".csv")
    for filepath in filepaths:
        exp_data = pd.read_csv(filepath)
        data.append(exp_data)
    return data
