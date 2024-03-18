# Last updated February 22, 2024
# Version 0.1.0
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd
import polars as pl

facecolor = sns.color_palette()[0]
emphasiscolor = sns.color_palette()[3]


def plot_missingness(
    df: pd.DataFrame | pl.DataFrame,
    size: Tuple[int, int] = None,
    title: str = "Percentage of Missing Values",
) -> None:
    if isinstance(df, pd.DataFrame):
        missing_per_column = (
            df.isnull().sum() / df.shape[0] * 100
        ).sort_values()
        missing_per_column.index.name = "Column"
    elif isinstance(df, pl.DataFrame):
        missing_per_column = (
            (df.null_count() / df.shape[0] * 100)
            .transpose(
                include_header=True,
                header_name="Column",
                column_names=["Missing"],
            )
            .to_pandas()
            .set_index("Column")
            .squeeze()
            .sort_values()
        )
    fig = plt.figure(figsize=size)
    ax = sns.barplot(
        x=missing_per_column.index,
        y=missing_per_column.values,
        color=facecolor,
    )
    ax.bar_label(
        ax.containers[-1],
        fmt=lambda x: f"{x:.0f}%" if round(x) > 0 else None,
        label_type="edge",
        size=8,
    )
    plt.xticks(rotation=90)
    plt.title(title)
    return


def plot_boxen_num(
    df: pd.DataFrame | pl.DataFrame,
    cols: List[str],
    hue: str,
    nrows: int,
    ncols: int,
    size: Tuple[int, int] = None,
    title: str = "Boxenplots of Numerical Variables",
) -> Tuple[Figure, Axes]:
    """Draw boxenplots of numerical features."""
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    if nrows * ncols > 1:
        axes = axes.flatten()
    for idx, col in enumerate(cols):
        if nrows * ncols == 1:
            ax = axes
        else:
            ax = axes[idx]
        sns.boxenplot(data=df, y=col, hue=hue, ax=ax, width=0.5)
        ax.set(ylabel=col, xlabel=None)
        if idx < len(cols) - 1:
            ax.get_legend().remove()
        else:
            sns.move_legend(ax, 6, bbox_to_anchor=(1, 0.5))
    for idx in range(len(cols), nrows * ncols):
        axes[idx].set_axis_off()
    fig.suptitle(title)
    return fig, axes


def plot_bar_cat(
    df: pd.DataFrame | pl.DataFrame,
    cols: List[str],
    hue: str,
    nrows: int,
    ncols: int,
    size: Tuple[int, int] = None,
    rotation: int = 0,
    title: str = "Barplots of Categorical Variables",
) -> Tuple[Figure, Axes]:
    """Draw barplots of numerical features."""
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    if nrows * ncols > 1:
        axes = axes.flatten()
    for idx, col in enumerate(cols):
        if nrows * ncols == 1:
            ax = axes
        else:
            ax = axes[idx]
        sns.histplot(
            data=df,
            x=col,
            hue=hue,
            ax=ax,
            stat="percent",
            common_norm=False,
            discrete=True,
            multiple="dodge",
            shrink=0.8,
        )
        for cont in ax.containers:
            ax.bar_label(
                cont,
                fmt=lambda x: f"{x:.0f}%" if round(x) > 0 else None,
                label_type="edge",
                size=8,
            )
        ax.set(ylabel=col, xlabel=None)
        ax.tick_params(axis="x", rotation=rotation, size=6)
        if idx < len(cols) - 1:
            ax.get_legend().remove()
        else:
            sns.move_legend(ax, 6, bbox_to_anchor=(1, 0.5))
    for idx in range(len(cols), nrows * ncols):
        axes[idx].set_axis_off()
    fig.suptitle(title)
    return fig, axes
