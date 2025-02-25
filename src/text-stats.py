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
import matplotlib.pyplot as plt

def main():
    # set lang
    textstat.set_lang("es")

    df = read_data()
    df = df.with_columns(
        fernandez_huerta=pl.col("content").map_elements(
            lambda x: textstat.fernandez_huerta(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        ),
        szigriszt_pazos=pl.col("content").map_elements(
            lambda x: textstat.szigriszt_pazos(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        )
    )

    # filter out everything that is not assistant 
    role = "assistant"
    df = df.filter(role = role)

    df = df.with_columns(total_message_number=pl.int_range(1, pl.len() + 1).over("id"))

    plot = df.plot.line(x="total_message_number", y="fernandez_huerta", color="group").properties(width=600, height=600)

    plot.encoding.y.title = "Fernandez Huerta Readability"
    plot.encoding.x.title = f"Message number ({role} only)"


    save_dir = Path(__file__).parents[1] / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot.save(save_dir / "fernandez_huerta_plot.html")

if __name__ == "__main__":
    main()
