"""
Script to extract all kinds of metrics for analysis

NB. Requires a "v{version}_dataset.csv" file in the data folder that can be created with create_dataset.py
"""

from pathlib import Path
from typing import Callable

import polars as pl
import textdescriptives as td
import textstat


def extract_td(
    df: pl.DataFrame,
    text_col_name: str = "content",
    spacy_model: str = "es_core_news_md",
    metrics_dir: Path = None,
    metrics_file_name: str = None,
) -> pl.DataFrame:
    """
    Extract TextDescriptives metrics
    """
    print(f"[INFO:] Extracting metrics using {spacy_model} and TextDescriptives")
    metrics_df = td.extract_metrics(text=df[text_col_name], spacy_model=spacy_model)

    # convert to polars
    metrics_df = pl.from_pandas(metrics_df)

    # drop col
    metrics_df = metrics_df.drop("text")

    # concat
    combined_df = pl.concat([df, metrics_df], how="horizontal")

    if metrics_dir is not None and metrics_file_name is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        combined_df.write_csv(metrics_dir / metrics_file_name)
    else:
        print(
            "[WARNING:] No metrics_dir or metrics_file_name provided. Metrics not saved."
        )

    return combined_df


def extract_textstat(
    df: pl.DataFrame,
    lang: str = "es",
    stats: list[Callable[[str], float]] = [
        textstat.fernandez_huerta,
        textstat.szigriszt_pazos,
        textstat.gutierrez_polini,
        textstat.crawford,
        textstat.flesch_kincaid_grade,
    ],
    metrics_dir: Path = None,
    metrics_file_name: str = None,
) -> pl.DataFrame:
    """
    Extract textstat metrics
    """
    print(f"[INFO:] Extracting metrics using textstat for language '{lang}'")
    for stat in stats:
        df = df.with_columns(
            pl.col("content")
            .map_elements(
                lambda x: stat(x) if isinstance(x, str) else None,
                return_dtype=pl.Float64,
            )
            .alias(stat.__name__)  # name column after function
        )

    if metrics_dir is not None and metrics_file_name is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        df.write_csv(metrics_dir / metrics_file_name)
    else:
        print(
            "[WARNING:] No metrics_dir or metrics_file_name provided. Metrics not saved."
        )

    return df

def main(): 
    data_path = Path(__file__).parents[1] / "data"
    version = 3.0

    # read data
    df = pl.read_csv(data_path / f"v{version}_dataset.csv")

    # define save paths
    metrics_dir = Path(__file__).parents[1] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # extract textdescriptives
    extract_td(
        df,
        metrics_dir=metrics_dir,
        metrics_file_name=f"v{version}_textdescriptives.csv",
    )

    # extract textstat
    extract_textstat(
        df,
        metrics_dir=metrics_dir,
        metrics_file_name=f"v{version}_text_stats.csv",
    )

if __name__ == "__main__":
    main()