"""
Plotting functions
"""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import polars as pl


def line_plot_variables(
    df: pl.DataFrame,
    x_var: str,
    y_vars: list[str],
    group_var: str = "group",
    model_var: str = "model",
    group_colors: list[str] = ["#008aff", "#ff471a", "#00a661"],
    x_label_text: str = None,
    ci_vars: list[str] = None  # Optional parameter for confidence intervals
) -> matplotlib.figure.Figure:
    # Ensure group_var is provided
    models = df[model_var].unique().to_list()
    groups = df[group_var].unique()

    # Prepare the color palette for the groups
    group_color_palette = {group: color for group, color in zip(groups, group_colors)}

    # Number of rows and columns for subplots
    n_rows = len(y_vars)
    n_cols = len(df[model_var].unique())

    # Create the figure and axes
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey="row"
    )

    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    # Loop through each y-variable and each model
    for i, y_var in enumerate(y_vars):
        for j, model in enumerate(models):
            ax = axes[i][j]

            # Filter the data for the current model
            model_data = df.filter(pl.col(model_var) == model)

            # Plot the data for each group
            for group in groups:
                group_data = model_data.filter(pl.col(group_var) == group)
                ax.plot(
                    group_data[x_var],
                    group_data[y_var],
                    label=group,
                    color=group_color_palette.get(group),
                )

                # Plot confidence interval if ci_vars is provided
                if ci_vars:
                        ci_lower = f"{ci_vars[i]}_lower"
                        ci_high = f"{ci_vars[i]}_high"
                        ax.fill_between(
                            group_data[x_var],
                            group_data[ci_lower],
                            group_data[ci_high],
                            color=group_color_palette.get(group),
                            alpha=0.1,  # Transparency of the CI fill
                        )

            # Set titles and labels
            if i == 0:
                ax.set_title(f"{model}", fontsize=14)

            if x_label_text and i == n_rows - 1:
                ax.set_xlabel(x_label_text)
            else:
                ax.set_xlabel("")  # explicitly remove labels from other rows

            if j == 0:
                ax.set_ylabel(" ".join(word.capitalize() for word in y_var.split("_")))

            if i == 0 and j == n_cols - 1:
                ax.legend(title=group_var.capitalize())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

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
    density_lines=False,
) -> plt.Figure:
    """
    Generates overlayed histogram or density plots to visualize data distribution across different groups and models.
    
    Parameters:
    - normalize (bool): If True, histograms are density plots (normalized). If False, absolute counts are used.
    - density_lines (bool): If True, draws density lines instead of histograms. Note: if density_lines=True, normalize has to be True.
    """
    if density_lines and not normalize:
        raise ValueError("Density lines are always normalized. Set normalize=True when using density_lines=True.")
    
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
                
                if density_lines:
                    sns.kdeplot(group_data, ax=ax, color=colors[k % len(colors)], label=group)
                else:
                    ax.hist(group_data, bins=bins, density=normalize, alpha=alpha, color=colors[k % len(colors)], label=group, edgecolor="white")
            
            if i == 0:
                ax.set_title(f"{model}")
            
            if j == 0:
                ax.set_ylabel("Density" if normalize else "Frequency")
            
            if x_label_texts:
                ax.set_xlabel(x_label_texts[i])
            
            if i == n_rows - 1 and j == n_cols - 1:
                ax.legend()
    
    fig.tight_layout()
    return fig

def scatter_with_regression(
    df: pl.DataFrame,
    x_vars: list[str],
    y_vars: list[str],
    group_var="group",
    model_var="model",
    colors: list[str] = ["#008aff", "#ff471a", "#00a661"],
    y_label_texts: list[str] = None,
) -> plt.Figure:
    """
    Generate scatter plots with regression lines to visualize data distribution across different groups and models.
    """
    models = df[model_var].unique().to_list()
    groups = df[group_var].unique().to_list()

    n_rows = len(x_vars)
    n_cols = len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey="row")

    # ensure axes is iterable even for single rows/columns
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, x_var in enumerate(x_vars):
        for j, model in enumerate(models):
            ax = axes[i][j]

            # filter data for current llm
            model_data = df.filter(pl.col(model_var) == model)

            # plot scatter and regression line for each group
            for k, group in enumerate(groups):
                group_data = model_data.filter(pl.col(group_var) == group)

                # get x and y values for plotting
                x_vals = group_data[x_var].to_numpy().flatten()
                y_vals = group_data[y_vars[i]].to_numpy().flatten()  # Use y_vars[i] to get correct y variable

                # scatter
                ax.scatter(x_vals, y_vals, color=colors[k % len(colors)], label=group, alpha=0.7)

                # reg line
                sns.regplot(x=x_vals, y=y_vals, ax=ax, scatter=False, color=colors[k % len(colors)], line_kws={"lw": 2})

            # title for first row 
            if i == 0:
                ax.set_title(f"{model}")

            # labels 
            ax.set_xlabel(x_var)

            # set y-axis label if provided
            if y_label_texts and j == 0:
                ax.set_ylabel(y_label_texts[i])
            else:
                ax.set_ylabel("")

            ax.legend(title=group_var)

    fig.tight_layout()
    return fig
