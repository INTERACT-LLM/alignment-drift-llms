"""
Quick conversion from ipynb to python script for plotting all plots for the paper
"""

from pathlib import Path

import polars as pl

from utils.plot_functions import distribution_plot, line_plot_variables
from utils.metrics_process import read_metrics, get_assistant_data, aggregate_df

def plot_readability(metrics_dir, plots_dir, model_dict, version, unique_models, pad_inches) -> None:
    """
    Plot textstat metrics
    """
    # define save path
    readability_dir = plots_dir / "readability"
    readability_dir.mkdir(parents=True, exist_ok=True)

    # read data, filter
    df = read_metrics(metrics_path=metrics_dir, model_dict=model_dict, version=version, metric_types=["text_stats"])
    assistant_df = get_assistant_data(df)

    # rename group columns
    assistant_df = assistant_df.rename({"group": "level"})
    
    # get aggregated data
    cols = ["fernandez_huerta", "szigriszt_pazos", "gutierrez_polini", "flesch_kincaid_grade", "crawford"]
    agg_df = aggregate_df(assistant_df, cols_to_aggregate=cols, ci_to_cols=True)

    # plotting
    colors = ["#008aff", "#ff471a", "#00a661"]

    cols = ["fernandez_huerta", "szigriszt_pazos", "gutierrez_polini"]
    vars = [f"{col}_mean" for col in cols]
    ci_vars = [f"{col}_ci" for col in cols]

    y_label_texts = ["Fernández Huerta", "Szigriszt-Pazos", "Gutiérrez de Polini"]

    # plot line plot
    fig = line_plot_variables(df=agg_df, 
                        x_var="total_message_number", 
                        y_vars=vars, 
                        ci_vars=ci_vars,
                        group_var="level",
                        model_var="model",
                        unique_models=unique_models,
                        x_label_text="Total Message Number", 
                        y_label_texts=y_label_texts,
                        y_lims={"fernandez_huerta_mean": (60, 115), "szigriszt_pazos_mean": (60, 115), "gutierrez_polini_mean": (30, 55)},
                        group_colors=colors)

    fig.savefig(readability_dir / "spanish_readability_high_easy_curves.png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    
    # plot distribution
    fig = distribution_plot(assistant_df, 
                        x_vars=["fernandez_huerta", "szigriszt_pazos", "gutierrez_polini"], 
                        group_var="level",
                        model_var="model",
                        unique_models=unique_models,
                        normalize=True,
                        x_label_texts=y_label_texts,
                        x_lims={"fernandez_huerta": (35, 160), "szigriszt_pazos": (35, 160), "gutierrez_polini": (10, 65)},
                        density_lines=True)

    fig.savefig(readability_dir / "spanish_readability_high_easy_dist.png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)


def plot_structural(metrics_dir, plots_dir, model_dict, version, unique_models, pad_inches) -> None:
    # define save path
    structural_dir = plots_dir / "structural"
    structural_dir.mkdir(parents=True, exist_ok=True)
    
    # read data, filter
    df = read_metrics(metrics_path=metrics_dir, model_dict=model_dict, version=version, metric_types=["textdescriptives"])
    assistant_df = get_assistant_data(df)

    # rename group columns
    assistant_df = assistant_df.rename({"group": "level"})

    # get aggregated data
    cols = ["doc_length", "dependency_distance_mean"]
    agg_df = aggregate_df(assistant_df, cols_to_aggregate=cols, ci_to_cols=True)

    # plotting
    colors = ["#008aff", "#ff471a", "#00a661"]
    cols = ["doc_length", "dependency_distance_mean"] # consider number of sentences
    vars = [f"{col}_mean" for col in cols]
    ci_vars = [f"{col}_ci" for col in cols]

    # plot line plot
    fig = line_plot_variables(df=agg_df, 
                    x_var="total_message_number", 
                    y_vars=vars,
                    ci_vars=ci_vars,
                    group_var="level",
                    model_var="model", 
                    unique_models=unique_models,
                    x_label_text="Total Message Number", 
                    y_label_texts=["Text Length", "Mean Dependency Distance"],
                    group_colors=colors)

    fig.savefig(structural_dir / "doc_length_and_MDD_curve.png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)

    # plot distribution
    fig = distribution_plot(assistant_df, 
                        x_vars=["doc_length", "dependency_distance_mean"], 
                        group_var="level",
                        model_var="model",
                        unique_models=unique_models,
                        normalize=True,
                        x_label_texts=["Text Length", "Mean Dependency Distance"], 
                        density_lines=True,
                        x_lims={"dependency_distance_mean": (0, 7)},
                        legend=False)

    fig.savefig(structural_dir / "doc_length_and_MDD_dist.png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)


def plot_surprisal(metrics_dir, plots_dir, model_dict, version, unique_models, pad_inches) -> None:
    # define save path
    surprisal_dir = plots_dir / "surprisal"
    surprisal_dir.mkdir(parents=True, exist_ok=True)

    # read data, filter
    df = read_metrics(metrics_path=metrics_dir, model_dict=model_dict, version=version, metric_types=["surprisal"])
    assistant_df = get_assistant_data(df)

    # rename to paragraph
    assistant_df = assistant_df.with_columns(
        pl.col("surprisal_mean").alias("surprisal_paragraph")
    )

    # rename group columns
    assistant_df = assistant_df.rename({"group": "level"})

    # get aggregated data
    cols = ["surprisal_paragraph"]

    # aggregate data
    agg_df = aggregate_df(assistant_df, cols_to_aggregate=cols, ci_to_cols=True)

    # plotting
    colors = ["#008aff", "#ff471a", "#00a661"]
    vars = [f"{col}_mean" for col in cols]
    ci_vars = [f"{col}_ci" for col in cols]

    # plot line plot
    fig = line_plot_variables(df=agg_df, 
                        x_var="total_message_number", 
                        y_vars=vars, 
                        ci_vars=ci_vars,
                        group_var="level",
                        model_var="model", 
                        unique_models=unique_models,
                        x_label_text="Total Message Number", 
                        y_label_texts=["Message Surprisal"],
                        group_colors=colors,
                        legend_position="upper right")

    fig.savefig(surprisal_dir / "surprisal_curve.png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)

    # plot distribution
    fig = distribution_plot(assistant_df, 
                            x_vars=["surprisal_paragraph"], 
                            normalize=True,
                            alpha=0.5, 
                            group_var="level",
                            model_var="model",
                            unique_models=unique_models,
                            x_label_texts=["Message Surprisal"], 
                            density_lines=True,
                            legend=False,
                            x_lims={"surprisal_paragraph": (0, 5)},
                            )

    fig.savefig(surprisal_dir / "surprisal_dist.png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)



def main():
    version = 3.0

    metrics_dir = Path(__file__).parents[1] / "metrics"
    plots_dir = Path(__file__).parents[1] / "plots" / f"v{version}"

    # for giving shorter names in df 
    model_dict = {
        "meta-llama--Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
        "google--gemma-3-12b-it": "Gemma 3 12B IT",
        "mistralai--Mistral-7B-Instruct-v0.3": "Mistral 7B Instruct v0.3",
        "Qwen--Qwen2.5-7B-Instruct": "Qwen 2.5 7B Instruct",
    }
    
    # setup
    unique_models = model_dict.values()
    pad_inches = 0.1

    # plot readability
    print("Plotting readability metrics...")
    plot_readability(metrics_dir, plots_dir, model_dict, version, unique_models, pad_inches)

    # plot structural
    print("Plotting structural metrics...")
    plot_structural(metrics_dir, plots_dir, model_dict, version, unique_models, pad_inches)

    # plot surprisal
    print("Plotting surprisal metrics...")
    plot_surprisal(metrics_dir, plots_dir, model_dict, version, unique_models, pad_inches)


if __name__ == "__main__":
    main()