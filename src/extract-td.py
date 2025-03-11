from pathlib import Path
import textdescriptives as td
import polars as pl


def main():
    data_path = Path(__file__).parents[1] / "data"
    version = 2.0

    df = pl.read_csv(data_path / f"v{version}_dataset.csv")
    metrics_df = td.extract_metrics(text=df["content"], spacy_model="es_core_news_md")

    # convert to polars
    metrics_df = pl.from_pandas(metrics_df)

    # drop col
    metrics_df = metrics_df.drop("text")

    # concat
    combined_df = pl.concat([df, metrics_df], how="horizontal")

    metrics_dir = Path(__file__).parents[1] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    combined_df.write_csv(metrics_dir / f"v{version}_textdescriptives.csv")

if __name__ == "__main__":
    main()
