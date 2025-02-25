from dataclasses import dataclass
from pathlib import Path

import polars as pl

@dataclass
class DataFile:
    folder_name: str
    file_name: str
    path: Path


def read_data(data_dir: Path) -> pl.DataFrame:
    """
    Read JSON data from path w. polars
    """

    datafiles = [
        DataFile(folder.name, file.name, str(file))
        for folder in data_dir.iterdir()
        if folder.is_dir()
        for file in folder.iterdir()
        if file.is_file() and file.suffix == ".json"
    ]

    dfs = []
    for file in datafiles:
        df = pl.read_json(file.path)
        # add cols

        df = df.with_columns(group=pl.lit(file.folder_name), id=pl.lit(file.file_name))
        dfs.append(df)

    combined_df = pl.concat(dfs)

    return combined_df