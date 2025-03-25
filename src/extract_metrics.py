"""
Script to extract all kinds of metrics for analysis

NB. Requires a "v{version}_dataset.csv" file in the data folder that can be created with create_dataset.py


"""

import argparse
from pathlib import Path
from typing import Callable, Literal, Optional

import polars as pl
import textdescriptives as td
import textstat
import statistics
import torch

from minicons import scorer

from utils.text_process import _split_text_into_sents


def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--metrics_pipeline",
        type=str,
        help="Specify which metrics to extract",
        choices=["textdescriptives", "textstats", "surprisal", "all"],
        default="all",
    )

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args


def extract_td(
    df: pl.DataFrame,
    text_col_name: str = "content",
    spacy_model: str = "es_core_news_md",
    metrics: list[str] = [
        "descriptive_stats",
        "dependency_distance",
        "pos_proportions",
        "quality",
        "coherence",
    ],
    metrics_dir: Path = None,
    metrics_file_name: str = None,
) -> pl.DataFrame:
    """
    Extract TextDescriptives metrics
    """
    print(f"[INFO:] Extracting metrics using {spacy_model} and TextDescriptives")
    metrics_df = td.extract_metrics(
        text=df[text_col_name], spacy_model=spacy_model, metrics=metrics
    )

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
    Extract textstat statistics
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


def extract_surprisal_sents(
    sents: list[str],
    model: Optional[scorer.IncrementalLMScorer] = None,
    model_id: Optional[str] = "gpt2",
    device: Optional[Literal["cpu", "cuda"]] = "cuda",
) -> list[float]:
    if model is None:
        # Default to "cuda" if no device is passed
        print(f"[INFO:] Loading model '{model_id}' on device '{device}'")
        model = scorer.IncrementalLMScorer(model_id, device)

    # compute surprisal in nats (normalized by number of tokens)
    surprisal_scores = model.sequence_score(
        sents, reduction=lambda x: -x.mean(0).item()
    )

    return surprisal_scores


def extract_surprisal(
    df: pl.DataFrame,
    text_col: str = "content",
    model_id: Optional[str] = "gpt2",
    device: Optional[Literal["cpu", "cuda"]] = "cuda",
    metrics_dir: Optional[Path] = None,
    metrics_file_name: Optional[str] = None,
) -> pl.DataFrame:
    """
    Extract surprisal metrics at the paragraph level.
    """

    # load model once to avoid reloading it for each row
    model = scorer.IncrementalLMScorer(model_id, device)

    # split text into sentences
    df = df.with_columns(
        sents=pl.col(text_col).map_elements(
            _split_text_into_sents, return_dtype=pl.List(pl.String)
        )
    )

    # compute surprisal per sentence
    print(f"[INFO:] Extracting surprisal metrics using model '{model_id}'")
    df = df.with_columns(
        # get surprisal scores for each sentence in "content"
        surprisal_per_sentence=pl.col("sents").map_elements(
            lambda x: extract_surprisal_sents(
                x, model=model
            ),  # Returns a list of floats
            return_dtype=pl.List(pl.Float64),
        )
    )

    # compute mean and median surprisal
    df = df.with_columns(
        surprisal_mean=pl.col("surprisal_per_sentence").map_elements(
            lambda x: statistics.mean(x), return_dtype=pl.Float64
        ),
        surprisal_median=pl.col("surprisal_per_sentence").map_elements(
            lambda x: statistics.median(x), return_dtype=pl.Float64
        ),
    )

    # convert sents and surprisal_per_sentence to strings (to save as csv)
    df = df.with_columns(
        sents=pl.col("sents").map_elements(
            lambda x: "[" + ", ".join(f'"{s}"' for s in x) + "]", return_dtype=pl.String
        ),
        surprisal_per_sentence=pl.col("surprisal_per_sentence").map_elements(
            lambda x: "[" + ", ".join(map(str, x)) + "]", return_dtype=pl.String
        ),
    )

    if metrics_dir is not None and metrics_file_name is not None:
        # saved_df = df.drop(["sents", "surprisal_per_sentence"]) # as polars cannot save list columns
        metrics_dir.mkdir(parents=True, exist_ok=True)
        df.write_csv(metrics_dir / metrics_file_name)

    return df


def main():
    args = input_parse()

    data_path = Path(__file__).parents[1] / "data"
    metrics_dir = Path(__file__).parents[1] / "metrics"
    version = 3.0

    # surprisal settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google-bert/bert-base-multilingual-cased"

    # read data
    df = pl.read_csv(data_path / f"v{version}_dataset.csv")

    if args.metrics_pipeline == "all":
        metrics_to_extract = ["surprisal", "textdescriptives", "textstats"]
    else:
        metrics_to_extract = [args.metrics_pipeline]


    for metric in metrics_to_extract:
        if metric == "textdescriptives":
            extract_td(
                df,
                metrics_dir=metrics_dir,
                metrics_file_name=f"v{version}_textdescriptives.csv",
            )
        elif metric == "textstats":
            extract_textstat(
                df,
                metrics_dir=metrics_dir,
                metrics_file_name=f"v{version}_text_stats.csv",
            )
        elif metric == "surprisal":
            extract_surprisal(
                df,
                device=device,
                model_id=model_id,
                metrics_dir=metrics_dir,
                metrics_file_name=f"v{version}_surprisal.csv",
            )
        else:
            print(["[ERROR:] Invalid metric pipeline."])

if __name__ == "__main__":
    main()
