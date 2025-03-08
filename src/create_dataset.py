from dataclasses import dataclass
from pathlib import Path

import polars as pl

@dataclass
class DataFile:
    dir_name: str
    folder_name: str
    file_name: str
    path: Path


def read_data(data_dir: Path | list[Path]) -> pl.DataFrame:
    """Read JSON data from path(s) with polars"""

    # Ensure we have a list of paths
    dirs_to_process = [data_dir] if isinstance(data_dir, Path) else data_dir
    
    datafiles = [
        DataFile(file.parents[2].name, folder.name, file.name, str(file))
        for directory in dirs_to_process
        for folder in directory.iterdir() if folder.is_dir()
        for file in folder.iterdir() if file.is_file() and file.suffix == ".json"
    ]

    print(f"[INFO:] Combining {len(datafiles)} files")
    
    if not datafiles:
        return pl.DataFrame(schema={"dir": pl.Utf8, "group": pl.Utf8, "id": pl.Utf8})

    dfs = []
    for file in datafiles:
        df = pl.read_json(file.path)
        # add cols

        df = df.with_columns(
                            model=pl.lit(file.dir_name),
                            group=pl.lit(file.folder_name), 
                            id=pl.lit(file.file_name)
                            )
        dfs.append(df)

    combined_df = pl.concat(dfs)

    return combined_df

def main(): 
    version = 2.0

    dir_names = ["mlx-community--Qwen2.5-7B-Instruct-1M-4bit", 
                 "mlx-community--meta-Llama-3.1-8B-Instruct-4bit"
                 ]

    data_paths = [
        Path(__file__).parents[1] / "data" / dir_name / f"v{version}"
        for dir_name in dir_names
    ]

    df = read_data(data_paths)
    df.write_csv(Path(__file__).parents[1] / "data" / f"v{version}_dataset.csv")

if __name__ == "__main__":
    main()

