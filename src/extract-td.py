from pathlib import Path
import textdescriptives as td
import polars as pl


def main():
    data_path = Path(__file__).parents[2] / "data"
    version = 2.0

    df = pl.read_csv(data_path / f"v{version}_dataset.csv")
    df = td.extract_metrics(text=df["content"], spacy_model="es_core_news_md")


    metrics_dir = Path(__file__).parents[1] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / f"v{version}_textdescriptives.csv")

if __name__ == "__main__":
    main()
