"""
extract readability metrics with text-stats
https://github.com/textstat/textstat

Interpretation: The lower, the more difficult
- Fernandez-Huerta (https://www.spanishreadability.com/fernandez-huerta-readability-index)
- Szigriszt-Pazos (https://www.spanishreadability.com/szigriszt-pazos-perspicuity-index)
"""

from pathlib import Path
import polars as pl
import textstat

def main():
    # set lang
    textstat.set_lang("es")

    data_path = (
        Path(__file__).parents[1]
        / "data"
    )   

    version = 2.0

    df = pl.read_csv(data_path / f"v{version}_dataset.csv")
    df = df.with_columns(
        fernandez_huerta=pl.col("content").map_elements(
            lambda x: textstat.fernandez_huerta(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        ),
        szigriszt_pazos=pl.col("content").map_elements(
            lambda x: textstat.szigriszt_pazos(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        ),
        flesch_reading_ease=pl.col("content").map_elements(
            lambda x: textstat.flesch_reading_ease(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        ),
        flesch_kincaid_grade=pl.col("content").map_elements(
            lambda x: textstat.flesch_kincaid_grade(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        )
    )
    
    metrics_dir = Path(__file__).parents[1] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(metrics_dir / f"v{version}_text_stats.csv")

    role = "assistant"
    df = df.filter(pl.col("role") == role)

    # Assign message numbers
    df = df.with_columns(
        total_message_number=pl.int_range(1, pl.len() + 1).over("id")
    )

    # Compute averages per group
    agg_df = df.group_by(["model", "group", "total_message_number"], maintain_order=True).agg(
        pl.col("fernandez_huerta").mean(),
        pl.col("flesch_reading_ease").mean(),
        pl.col("flesch_kincaid_grade").mean()
    )

    # Compute total unique IDs per model (overall counts)
    model_n = df.group_by("model").agg(n=pl.col("id").n_unique())

    # Map of model to total n
    model_n_dict = dict(zip(model_n["model"], model_n["n"]))

    vars = ["fernandez_huerta", "flesch_reading_ease", "flesch_kincaid_grade"]

    for var in vars:
        for model_name in df["model"].unique():
            model_df = agg_df.filter(pl.col("model") == model_name)

            total_n = model_n_dict[model_name]

            plot = (
                model_df.plot.line(
                    x="total_message_number",
                    y=var,
                    color="group"
                )
                .properties(
                    width=600,
                    height=600,
                    title=f"Average Score for {model_name} (n = {total_n})"
                )
                .configure_scale(zero=False)
            )

            plot.encoding.y.title = ' '.join([word.capitalize() for word in var.split('_')])
            plot.encoding.x.title = f"Message number ({role} only)"

            save_dir = Path(__file__).parents[1] / "plots"
            save_dir.mkdir(parents=True, exist_ok=True)
            plot.save(save_dir / f"{model_name}_v{version}_{var}_plot.html")


if __name__ == "__main__":
    main()
