from dataclasses import dataclass
from pathlib import Path

import polars as pl


DIR_NAMES = [
            "google--gemma-3-12b-it",
            "meta-llama--Llama-3.1-8B-Instruct", 
            "mistralai--Mistral-7B-Instruct-v0.3",
            "Qwen--Qwen2.5-7B-Instruct",
]

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
    
    # sort folders (keep consistent order)
    datafiles = [
        DataFile(file.parents[2].name, folder.name, file.name, str(file))
        for directory in dirs_to_process
        for folder in sorted(directory.iterdir()) if folder.is_dir()
        for file in sorted(folder.iterdir()) if file.is_file() and file.suffix == ".json"
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
    version = 3.0

    data_paths = [
        Path(__file__).parents[1] / "data" / dir_name / f"v{version}"
        for dir_name in DIR_NAMES
    ]

    df = read_data(data_paths)
    df.write_csv(Path(__file__).parents[1] / "data" / f"v{version}_dataset.csv")

if __name__ == "__main__":
    main()

