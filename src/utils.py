from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import polars as pl

DEFAULT_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "mlx-community--Qwen2.5-7B-Instruct-1M-4bit"
    / "v1.0"
)


@dataclass
class DataFile:
    folder_name: str
    file_name: str
    path: Path


def read_data(data_dir: Path = DEFAULT_PATH) -> pl.DataFrame:
    """
    Read JSON data from path w. polars
    """

    datafiles = [
        DataFile(folder.name, file.name, str(file))
        for folder in data_dir.iterdir()
        if folder.is_dir()
        for file in folder.iterdir()
        if file.is_file()
    ]

    dfs = []
    for file in datafiles:
        df = pl.read_json(file.path)
        # add cols

        df = df.with_columns(group=pl.lit(file.folder_name), id=pl.lit(file.file_name))
        dfs.append(df)

    combined_df = pl.concat(dfs)

    return combined_df
