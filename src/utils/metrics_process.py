"""
Preprocess metrics data
"""

from pathlib import Path
import polars as pl
from polars.testing import assert_frame_equal

# aka prettier names for plots
MODEL_DICT = {
    "mlx-community--Qwen2.5-7B-Instruct-1M-4bit": "Qwen 2.5 7B Instruct 1M (4bit)",
    "mlx-community--meta-Llama-3.1-8B-Instruct-4bit": "Llama 3.1 8B Instruct (4bit)",
    "meta-llama--Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "Qwen--Qwen2.5-7B-Instruct": "Qwen 2.5 7B Instruct",
    "mistralai--Mistral-7B-Instruct-v0.3": "Mistral 7B Instruct v0.3",
}

def read_metrics(
    metrics_path=Path(__file__).parents[1] / "metrics",
    metric_types=["surprisal", "text_stats", "textdescriptives"],
    version: float = 3.0,
    group_levels = ["A1", "B1", "C1"],
    model_dict = MODEL_DICT
):
    """
    Custom function to read metrics data from data/metrics

    Args:
    metrics_path: Path to the metrics data
    metric_types: List of metric types to read (i.e., the three file types "text_stats", "textdescriptives", ""surprisal)
        Will be combined into a single dataframe if multiple types are provided
    version: Version of the metrics data to read
    """
    dfs = []
    levels = pl.Enum(group_levels)
    for metric_type in metric_types:
        df = pl.read_csv(metrics_path / f"v{version}_{metric_type}.csv",
                         schema_overrides={"group": levels})
        
        # sort ids
        df = df.sort("id")

        dfs.append(df)

    # smash columns together (align method to not get duplicate columns)
    if len(dfs) > 1:
        combined_df = pl.concat(dfs, how="align_inner")
        combined_df = dfs[0]

    # replace model names with prettier names
    if model_dict:
        combined_df = combined_df.with_columns(pl.col("model").replace_strict(model_dict))

    return combined_df

# FILTER FNS
def filter_metrics(df, filter_col = "role", filter_val = "assistant"):
    """
    Filter metrics data by a specific value in a column

    Args:
        df: Metrics data
        filter_col: Column to filter by
        filter_val: Value to filter by
    """
    filtered_df = df.filter(pl.col(filter_col) == filter_val)
    return filtered_df

def compute_total_message_number(df):
    """
    Counting the total number of messages. 
    I usually apply after filtering to only one role (but could in principle be applied to a full user-assistant dataset)
    """
    new_df = df.with_columns(total_message_number=pl.int_range(1, pl.len() + 1).over("id"))    
    return new_df

## AGG FNS
def add_ci_to_col(df, col):
    """
    Add ci, ci_low, cgi_high columns to Polars DataFrame
    """
    # First calculate the CI column
    df = df.with_columns(
        (1.96 * pl.col(f"{col}_std") / (pl.col(f"{col}_count").sqrt())).alias(f"{col}_ci")
    )
    
    # Now calculate the low and high CI bounds using the already computed CI
    df = df.with_columns(
        (pl.col(f"{col}_mean") - pl.col(f"{col}_ci")).alias(f"{col}_ci_lower"),
        (pl.col(f"{col}_mean") + pl.col(f"{col}_ci")).alias(f"{col}_ci_high"),
    )
    
    return df

def aggregate_df(df, 
                 group_by_vars=["model", "group", "total_message_number"],
                 cols_to_aggregate=["fernandez_huerta", "szigriszt_pazos", "gutierrez_polini", "flesch_kincaid_grade", "crawford"],
                 ci_to_cols: bool = True,
                 ):
    """
    Group by variables and aggregate metrics data
    """
    agg_df = (
        df.group_by(group_by_vars, maintain_order=True)
        .agg([
            *[pl.col(col).mean().alias(f"{col}_mean") for col in cols_to_aggregate],
            *[pl.col(col).std().alias(f"{col}_std") for col in cols_to_aggregate],
            *[pl.col(col).count().alias(f"{col}_count") for col in cols_to_aggregate],
        ])
    )

    if ci_to_cols:
        for col in cols_to_aggregate:
            agg_df = add_ci_to_col(agg_df, col)

    return agg_df

def get_assistant_data(df):
    """
    Get only assistant data
    """
    filtered_df = filter_metrics(df)
    filtered_df = compute_total_message_number(filtered_df)

    return filtered_df

if __name__ == "__main__":
    df = read_metrics(metrics_path=Path(__file__).parents[2] / "metrics", metric_types=["text_stats", "textdescriptives"])

    print(len(df))

    assistant_df = get_assistant_data(df)

    agg_df = aggregate_df(assistant_df)

