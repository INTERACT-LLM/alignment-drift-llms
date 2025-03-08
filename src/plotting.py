import altair as alt
import polars as pl
from pathlib import Path



def main(): 
    # read data 
    metrics_path = (
            Path(__file__).parents[1]
            / "metrics"
        )   

    version = 2.0

    df = pl.read_csv(metrics_path / f"v{version}_text_stats.csv")
    role = "assistant"
    df = df.filter(pl.col("role") == role)

    # assign msg numbers
    df = df.with_columns(
        total_message_number=pl.int_range(1, pl.len() + 1).over("id")
    )

    # Filter for assistant role
    role = "assistant"
    df = df.filter(pl.col("role") == role)
    df = df.with_columns(total_message_number=pl.int_range(1, pl.len() + 1).over("id"))

    # Compute average scores
    avg_df = df.group_by(["model", "group", "total_message_number"], maintain_order=True).mean()

    # Define variables to plot
    vars = ["fernandez_huerta", "flesch_reading_ease", "flesch_kincaid_grade"]

    # Initialize an empty list to hold plots
    plots = []

    # Loop over variables and models
    for var in vars:
        row_plots = []
        for model_name in df["model"].unique():
            model_df = avg_df.filter(pl.col("model") == model_name)

            chart = (alt.Chart(model_df)
                    .mark_line()
                    .encode(
                        x=alt.X("total_message_number", title=f"Message number ({role} only)"),
                        y=alt.Y(var, scale=alt.Scale(zero=False), title=' '.join(word.capitalize() for word in var.split('_'))),
                        color=alt.Color("group", title="Group")
                    )
                    .properties(
                        width=500,
                        height=250,
                        title=f"{model_name}"
                    ))

            row_plots.append(chart)

        # Combine model plots horizontally per variable
        combined_row = alt.hconcat(*row_plots).resolve_scale(y='shared')
        combined_row = combined_row.properties(title=f"{var.replace('_', ' ').title()}")
        plots.append(combined_row)

    # Combine all variable plots vertically
    final_plot = alt.vconcat(*plots).configure_concat(spacing=20)
    final_plot = final_plot.configure_title(fontSize=20, offset=5, orient='top', anchor='middle')

    # Save the final combined plot
    save_dir = Path(__file__).parents[1] / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    final_plot.save(save_dir / f"combined_v{version}_model_comparison.html")


if __name__ == "__main__":
    main()