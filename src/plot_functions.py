"""
Plotting functions
"""

import polars as pl
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt


def line_plot_variables(
    df: pl.DataFrame,
    x_var: str,
    y_vars: list[str],
    group_var: str = "group",
    model_var: str = "model",
    group_colors: list[str] = ["#008aff", "#ff471a", "#00a661"],
    x_label_text: str = None,
) -> matplotlib.figure.Figure:
    models = df[model_var].unique() if model_var else [None]
    groups = df[group_var].unique() if group_var else [None]

    group_color_palette = (
        {group: color for group, color in zip(groups, group_colors)}
        if group_var
        else {}
    )

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

            data_filter = (pl.col(model_var) == model) if model_var else True
            current_df = df.filter(data_filter)

            if group_var:
                for group in groups:
                    group_data = current_df.filter(pl.col(group_var) == group)
                    ax.plot(
                        group_data[x_var],
                        group_data[y_var],
                        label=group,
                        color=group_color_palette.get(group),
                    )
            else:
                ax.plot(
                    current_df[x_var], current_df[y_var], label=model if model else None
                )
            
            if i == 0: 
                ax.set_title(f"{model}" if model else "", fontsize=14)

            if x_label_text and i == n_rows - 1:
                ax.set_xlabel(x_label_text)
            else:
                ax.set_xlabel("")  # explicitly remove labels from other rows
            
            if j == 0:
                ax.set_ylabel(" ".join(word.capitalize() for word in y_var.split("_")))

            if group_var and (i == 0 and j == n_cols - 1):
                ax.legend(title=group_var.capitalize())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def bar_plot(
    df: pl.DataFrame,
    x_vars: list[str],
    group_var="group",
    model_var="model",
    colors: list[str] = ["#008aff", "#ff471a", "#00a661"],
    std_vars: list[str] = None,
    y_label_texts: list[str] = None,
) -> matplotlib.figure.Figure:
    # get unique models and groups
    models = df[model_var].unique().to_list()
    groups = df[group_var].unique().to_list()

    n_rows = len(x_vars)
    n_cols = len(models)

    # bar settings
    width = 0.65
    x = np.arange(len(groups))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey="row")

    # ensure axes is iterable even for single rows/columns
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, x_var in enumerate(x_vars):
        for j, model in enumerate(models):
            ax = axes[i][j]

            model_data = df.filter(pl.col(model_var) == model)

            proportions = [
                model_data.filter(pl.col(group_var) == group)[x_var][0]
                for group in groups
            ]

            if std_vars:
                std_devs = [
                    model_data.filter(pl.col(group_var) == group)[std_vars[i]][0]
                    for group in groups
                ]
                ax.bar(x, proportions, width=width, color=colors, yerr=std_devs, capsize=5)
            else:
                ax.bar(x, proportions, width=width, color=colors)

            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            
            if i == 0:
                ax.set_title(f"{model}")

            if y_label_texts and j == 0:
                ax.set_ylabel(y_label_texts[i])
            else:
                ax.set_ylabel("")

    fig.tight_layout()

    return fig

def violin_plot(
    df: pl.DataFrame,
    x_vars: list[str],
    group_var="group",
    model_var="model",
    colors: list[str] = ["#008aff", "#ff471a", "#00a661"],
    y_label_texts: list[str] = None,
) -> matplotlib.figure.Figure:
    """
    Generates violin plots to visualize data distribution across different groups and models.
    If compute_error_bars is True, adds error bars representing standard deviation.
    """
    # get unique models and groups
    models = df[model_var].unique().to_list()
    groups = df[group_var].unique().to_list()

    n_rows = len(x_vars)
    n_cols = len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey="row")

    # ensure axes is iterable even for single rows/columns
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, x_var in enumerate(x_vars):
        for j, model in enumerate(models):
            ax = axes[i][j]

            model_data = df.filter(pl.col(model_var) == model)

            # extract group data 
            violin_data = [
                model_data.filter(pl.col(group_var) == group)[x_var].to_list()
                for group in groups
            ]

            parts = ax.violinplot(violin_data, showmeans=True)

            # set colors 
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.9)

            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)
            
            # Set x-axis labels
            ax.set_xticks(np.arange(1, len(groups) + 1))
            ax.set_xticklabels(groups)

            # Set title for the first row
            if i == 0:
                ax.set_title(f"{model}")

            # Set y-axis labels if provided
            if y_label_texts and j == 0:
                ax.set_ylabel(y_label_texts[i])
            else:
                ax.set_ylabel("")

    fig.tight_layout()
    return fig

def distribution_plot(
    df: pl.DataFrame,
    x_vars: list[str],
    group_var="group",
    model_var="model",
    colors: list[str] = ["#008aff", "#ff471a", "#00a661"],
    bins=30,
    alpha=0.6,
    normalize=True,
    x_label_texts: list[str] = None,
) -> matplotlib.figure.Figure:
    """
    Generates overlayed histogram plots to visualize data distribution across different groups and models.
    
    Parameters:
    - normalize (bool): If True, histograms are density plots (normalized). If False, absolute counts are used.
    """
    models = df[model_var].unique().to_list()
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
                ax.hist(group_data, bins=bins, density=normalize, alpha=alpha, color=colors[k % len(colors)], label=group,  edgecolor="white")
            
            if i == 0:
                ax.set_title(f"{model}")
            
            if j == 0:
                if normalize == True:
                    ax.set_ylabel("Density")
                elif normalize == False:
                    ax.set_ylabel("Frequency")
            
            if x_label_texts: 
                ax.set_xlabel(x_label_texts[i])

            if i == n_rows - 1 and j == n_cols - 1:
                ax.legend()
    
    fig.tight_layout()
    return fig