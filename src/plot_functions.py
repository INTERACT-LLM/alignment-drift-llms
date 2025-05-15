"""
Plotting functions
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import polars as pl


import matplotlib.pyplot as plt
import polars as pl

def line_plot_variables(
    df: pl.DataFrame,
    x_var: str,
    y_vars: list[str],
    group_var: str = "group",
    model_var: str = "model",
    unique_models: list[str] = None,
    group_colors: list[str] = ["#008aff", "#ff471a", "#00a661"],
    x_label_text: str = None,
    y_label_texts: list[str] = None,
    ci_vars: list[str] = None,
    y_lims: dict[str, tuple[float, float]] = None,
    legend: bool = True,
    legend_position: str = "best", 
) -> plt.Figure:
    if unique_models is None:
        print("[INFO:] No unique models provided. Using all models in the dataframe.")
        models = df[model_var].unique().to_list()
    else:
        models = unique_models

    groups = df[group_var].unique()
    group_color_palette = {group: color for group, color in zip(groups, group_colors)}
    n_rows = len(y_vars)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey="row"
    )

    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, y_var in enumerate(y_vars):
        for j, model in enumerate(models):
            ax = axes[i][j]
            model_data = df.filter(pl.col(model_var) == model)

            for group in groups:
                group_data = model_data.filter(pl.col(group_var) == group)
                ax.plot(
                    group_data[x_var],
                    group_data[y_var],
                    label=group,
                    color=group_color_palette.get(group),
                )
                if ci_vars:
                    ci_lower = f"{ci_vars[i]}_lower"
                    ci_high = f"{ci_vars[i]}_high"
                    ax.fill_between(
                        group_data[x_var],
                        group_data[ci_lower],
                        group_data[ci_high],
                        color=group_color_palette.get(group),
                        alpha=0.1,
                    )

            if y_lims and y_var in y_lims:
                ax.set_ylim(y_lims[y_var])
            if i == 0:
                ax.set_title(f"{model}", fontsize=16)
            if x_label_text and i == n_rows - 1:
                ax.set_xlabel(x_label_text, fontsize=14)
            else:
                ax.set_xlabel("")
            if j == 0 and y_label_texts:
                ax.set_ylabel(y_label_texts[i], fontsize=14)

            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="gray", alpha=0.3)

            # Legend on last top-right plot
            if legend and i == 0 and j == n_cols - 1:
                ax.legend(
                    title=group_var.capitalize(),
                    fontsize=13,
                    title_fontsize=13,
                    loc=legend_position,  
                    frameon=True,
                )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def distribution_plot(
    df: pl.DataFrame,
    x_vars: list[str],
    group_var="group",
    model_var="model",
    unique_models: list[str] = None,
    group_colors: list[str] = ["#008aff", "#ff471a", "#00a661"],
    bins=30,
    alpha=0.6,
    normalize=True,
    x_label_texts: list[str] = None,
    x_lims: dict[str, tuple[float, float]] = None,
    density_lines=False,
    legend=True,
) -> plt.Figure:
    """
    Generates overlayed histogram or density plots to visualize data distribution across different groups and models.
    """
    if density_lines and not normalize:
        raise ValueError("Density lines are always normalized. Set normalize=True when using density_lines=True.")
    
    if unique_models is None:
        print("[INFO:] No unique models provided. Using all models in the dataframe.")
        models = df[model_var].unique().to_list()
    else: 
        models = unique_models
    
    groups = df[group_var].unique().to_list()
    
    n_rows = len(x_vars)
    n_cols = len(models)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey="row")
    
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, x_var in enumerate(x_vars):
        for j, model in enumerate(models):
            ax = axes[i][j]
            model_data = df.filter(pl.col(model_var) == model)
            
            for k, group in enumerate(groups):
                group_data = model_data.filter(pl.col(group_var) == group)[x_var].to_list()
                
                if density_lines:
                    sns.kdeplot(group_data, ax=ax, color=group_colors[k % len(group_colors)], label=group)
                else:
                    ax.hist(
                        group_data,
                        bins=bins,
                        density=normalize,
                        alpha=alpha,
                        color=group_colors[k % len(group_colors)],
                        label=group,
                        edgecolor="white"
                    )
            
            # set axis title and labels
            if i == 0:
                ax.set_title(f"{model}", fontsize=16)
            if j == 0:
                ax.set_ylabel("Density" if normalize else "Frequency", fontsize=14)
            if x_label_texts:
                ax.set_xlabel(x_label_texts[i], fontsize=14)

            if x_lims and x_var in x_lims:
                ax.set_xlim(x_lims[x_var])

            # increase tick font size
            ax.tick_params(axis="both", which="major", labelsize=12)

            # add soft major grid
            ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="gray", alpha=0.3)
            
            # add legend in last top-right plot
            if legend and i == 0 and j == n_cols - 1:
                ax.legend(title=group_var.capitalize(), fontsize=13, title_fontsize=13)
    
    fig.tight_layout()
    return fig
