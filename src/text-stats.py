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


from utils import read_data


def main():
    # set lang
    textstat.set_lang("es")

    version = 2.0
    
    data_path = (
        Path(__file__).parents[1]
        / "data"
        / "mlx-community--Qwen2.5-7B-Instruct-1M-4bit"
        / f"v{version}"
    )   

    df = read_data(data_dir=data_path)
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
        )
    )

    # filter out everything that is not assistant 
    role = "assistant"
    df = df.filter(role = role)
    df = df.with_columns(total_message_number=pl.int_range(1, pl.len() + 1).over("id"))
    avg_df = df.group_by(["group", "total_message_number"], maintain_order=True).mean()

    vars = ["fernandez_huerta", "flesch_reading_ease"]

    for var in vars:
        plot = (avg_df.plot.line(
                x="total_message_number", 
                y=var, 
                color="group"
                )
                .properties(width=600, height=600, title=f"Average Score (n = 6)")
                .configure_scale(zero=False)
        )

        plot.encoding.y.title = f"{" ".join([word.capitalize() for word in var.split("_")])}"
        plot.encoding.x.title = f"Message number ({role} only)"

        save_dir = Path(__file__).parents[1] / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)
        plot.save(save_dir / f"v{version}_{var}_plot.html")

if __name__ == "__main__":
    main()
