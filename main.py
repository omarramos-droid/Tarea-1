import pandas as pd
from lib import lib

from src.config import data_dir

if __name__ == "__main__":
    # The filename that contains our data, once is downloaded
    data_filename = data_dir / "raw" / "2023-002_ISONET-Project-Members_13C_Data.txt"
    # The correct way to load the data
    txt = pd.read_csv(data_filename, encoding='ANSI', sep=None, header=3, engine="python")

    print(txt)