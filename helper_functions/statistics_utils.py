# Last updated February 20, 2024
# Version 0.1.0


from typing import Tuple, List
import pandas as pd
import numpy as np
import phik
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import kruskal, f_oneway, chi2_contingency


def difference_ind_samples_num(
    df: pd.DataFrame,
    num_cols: List[str],
    target_col: str,
    alpha: float = 0.05,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Perform one-way ANOVA and Kruskal-Wallis tests for independent samples
    to check if distributions of numerical features between samples of
    different targets are statistically significantly different. The one-way
    ANOVA tests the null hypothesis that two or more groups have the same
    population mean. The Kruskal-Wallis H-test tests the null hypothesis that
    the population median of all of the groups are equal.
    ANOVA has assumption that each sample is from a normally distributed
    population. The Kruskal-Wallis H-test tests is a non-parametric version of
    ANOVA. Returns dataframe with column names, p-values and indication if
    samples is statistically different based on both test (null hypothesis
    rejected with p-value less than significance level alpha in both tests).
    Also return a list of column names to drop."""
    result = pd.DataFrame()
    drop_features = []
    for col in num_cols:
        samples = (
            df[[target_col, col]].dropna().groupby(target_col)[col].apply(list)
        )
        _, p_kw = kruskal(*samples)
        _, p_owa = f_oneway(*samples)
        result.loc[col, "Kruskal-Wallis p-value"] = round(p_kw, 3)
        result.loc[col, "One-way ANOVA p-value"] = round(p_owa, 3)
        if (p_kw < alpha) & (p_owa < alpha):
            result.loc[col, "Significantly different"] = 1
        else:
            result.loc[col, "Significantly different"] = 0
            drop_features.append(col)
    if verbose:
        print(
            "Statistically unrelated to target numerical features:"
            f" {drop_features}"
        )
    return (result, drop_features)


def difference_ind_samples_cat(
    df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    alpha: float = 0.05,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Chi-square test of independence of variables in a contingency table.
    This function computes the chi-square p-value for the hypothesis test of
    independence of the observed frequencies in the contingency table. The null
    hypothesis is that two features are unrelated. Returns dataframe with column
    names, p-values and indication if samples is statistically independent (null
    hypothesis rejected with p-value less than significance level alpha).
    Also return a list of column names to drop due to nonsignificant relation
    to target."""
    result = pd.DataFrame()
    drop_features = []
    for col in cat_cols:
        contingency_table = pd.crosstab(df[target_col], df[col])
        res = chi2_contingency(contingency_table)
        result.loc[col, "Chi-square p-value"] = round(res.pvalue, 3)
        if res.pvalue < alpha:
            result.loc[col, "Significantly dependent"] = 1
        else:
            result.loc[col, "Significantly dependent"] = 0
            drop_features.append(col)
    if verbose:
        print(
            "Statistically unrelated to target categorical features:"
            f" {drop_features}"
        )
    return (result, drop_features)


def draw_phik(
    df: pd.DataFrame,
    num_features: List[str] | None = None,
    title: str = None,
    size: Tuple[int, int] = None,
) -> None:
    """Return phik correlation matrix of the provided pandas dataframe"""
    phik_corr = df.phik_matrix(interval_cols=num_features)
    mask = np.triu(np.ones_like(phik_corr, dtype=bool))
    fig = plt.figure(figsize=size)
    sns.heatmap(
        phik_corr,
        mask=mask,
        annot=True,
        cmap="RdYlGn",
        fmt=".2f",
        annot_kws={"size": 8},
        vmin=0,
        vmax=1,
        linewidths=1,
        xticklabels=True,
        yticklabels=True,
    )
    plt.grid(False)
    if title:
        plt.title(f"$\phi_K$ Correlation of {title}")
    else:
        plt.title(f"$\phi_K$ Correlation")
    plt.show()
    return


def phik_above(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.05,
    num_features: List[str] | None = None,
) -> List[str]:
    """Function to return columns correlating with indicated target with phik
    correlation coefficient with at least indicated threshold."""
    phik_corr = df.phik_matrix(interval_cols=num_features)
    corr_cols = [
        col for col in phik_corr.columns if phik_corr[col][target] >= threshold
    ]
    return corr_cols
