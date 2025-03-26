"""
Script to extract all kinds of metrics for analysis

NB. Requires a "v{version}_dataset.csv" file in the data folder that can be created with create_dataset.py


"""
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Callable, Literal, Optional

import polars as pl
import textdescriptives as td
import textstat
import statistics
import torch
import math

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
    model: Optional[scorer.MaskedLMScorer] = None,
    model_id: Optional[str] = "gpt2",
    device: Optional[Literal["cpu", "cuda", "auto"]] = "auto",
    batch_size: Optional[int] = None
) -> list[float]:
    """
    Extract surprisal scores for a list of sentences with improved memory management.
    
    Args:
        sents (list[str]): List of sentences to process
        model (Optional[scorer.MaskedLMScorer]): Pre-loaded model
        model_id (str): Model identifier
        device (str): Device to run the model on
        batch_size (Optional[int]): Batch size for processing
    
    Returns:
        list[float]: Surprisal scores for input sentences
    """
    if model is None:
        print(f"[INFO:] Loading model '{model_id}' on device '{device}'")
        model = scorer.MaskedLMScorer(
            model_id, 
            device, 
            trust_remote_code=True, 
            torch_dtype=torch.float16,
            batch_size=batch_size or len(sents)
        )

    # process all sentences in one go or in specified batch size
    batch_size = batch_size or len(sents)
    all_surprisal_scores = []
    
    for i in range(0, len(sents), batch_size):
        batch = sents[i:i + batch_size]
        
        try:
            with torch.no_grad():
                torch.cuda.empty_cache()  # clear mem before processing
                surprisal_scores = model.sequence_score(
                    batch, 
                    reduction=lambda x: -x.mean(0).item()
                )
            all_surprisal_scores.extend(surprisal_scores)
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            # fallback to processing individual sentences if batch fails (not really helpful if batch_size is 1)
            for sent in batch:
                try:
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        score = model.sequence_score(
                            [sent], 
                            reduction=lambda x: -x.mean(0).item()
                        )[0]
                    all_surprisal_scores.append(score)
                except Exception as e:
                    print(f"Error processing sentence: {sent}")
                    all_surprisal_scores.append(float('nan'))
        
        # clear mem after each batch
        del batch
        torch.cuda.empty_cache()

    return all_surprisal_scores


def extract_surprisal(
    df: pl.DataFrame,
    text_col: str = "content",
    model_id: Optional[str] = "gpt2",
    device: Optional[Literal["cpu", "cuda", "auto"]] = "auto",
    metrics_dir: Optional[Path] = None,
    metrics_file_name: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> pl.DataFrame:
    """
    Extract surprisal metrics at the paragraph level with improved error handling.
    
    Args:
        df (pl.DataFrame): Input DataFrame
        text_col (str): Column containing text
        model_id (str): Model identifier
        device (str): Device to run the model on
        metrics_dir (Optional[Path]): Directory to save metrics
        metrics_file_name (Optional[str]): Filename for metrics
        batch_size (Optional[int]): Batch size for processing
    
    Returns:
        pl.DataFrame: DataFrame with surprisal metrics
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() and device == "auto" else device

    # Load model once to avoid reloading it for each row
    model = scorer.MaskedLMScorer(
        model_id, 
        device, 
        trust_remote_code=True, 
        torch_dtype=torch.float16
    )

    # Preprocess text into sentences
    df = df.with_columns(
        sents=pl.col(text_col).map_elements(
            _split_text_into_sents, 
            return_dtype=pl.List(pl.String)
        )
    )

    # manual processing
    sents_list = df["sents"].to_list()
    surprisal_scores = []
    print(f"[INFO:] Extracting surprisal metrics using model '{model_id}'")
    for sents in tqdm(sents_list, desc="Computing surprisal"):
        try:
            batch_scores = extract_surprisal_sents(
                sents, 
                model=model, 
                batch_size=batch_size
            )
            surprisal_scores.append(batch_scores)
        except Exception as e:
            print(f"Error processing sentences: {e}")
            # append empty scores
            surprisal_scores.append([float('nan')] * len(sents))

    # add surprisal scores
    df = df.with_columns(
        pl.Series(name="surprisal_per_sentence", values=surprisal_scores)
    )

    # compute mean and median surprisal with error handling
    print("[INFO:] Computing means, medians")
    df = df.with_columns(
        surprisal_mean=pl.col("surprisal_per_sentence").map_elements(
            lambda x: statistics.mean([v for v in x if not math.isnan(v)]) if any(not math.isnan(v) for v in x) else float('nan'), 
            return_dtype=pl.Float64
        ),
        surprisal_median=pl.col("surprisal_per_sentence").map_elements(
            lambda x: statistics.median([v for v in x if not math.isnan(v)]) if any(not math.isnan(v) for v in x) else float('nan'),
            return_dtype=pl.Float64
        ),
    )

    # Convert sentences and surprisal scores to storable format
    df = df.with_columns(
        sents=pl.col("sents").map_elements(
            lambda x: "[" + ", ".join(f'"{s}"' for s in x) + "]", 
            return_dtype=pl.String
        ),
        surprisal_per_sentence=pl.col("surprisal_per_sentence").map_elements(
            lambda x: "[" + ", ".join(map(str, x)) + "]", 
            return_dtype=pl.String
        ),
    )

    # Save results if directories are provided
    if metrics_dir is not None and metrics_file_name is not None:
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
    model_id = "EuroBERT/EuroBERT-210m"

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
