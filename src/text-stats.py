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
        # spanish specific metrics
        fernandez_huerta=pl.col("content").map_elements(
            lambda x: textstat.fernandez_huerta(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        ),
        szigriszt_pazos=pl.col("content").map_elements(
            lambda x: textstat.szigriszt_pazos(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        ),
    
        gutierrez_polini=pl.col("content").map_elements(
            lambda x: textstat.gutierrez_polini(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        ),
        crawford=pl.col("content").map_elements(
            lambda x: textstat.crawford(x) if isinstance(x, str) else None,
            return_dtype=pl.Float64,
        ),
        
        # formulas originally for english, but computed with spanish syllables (due to textstat.set_lang("es"))
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

if __name__ == "__main__":
    main()
