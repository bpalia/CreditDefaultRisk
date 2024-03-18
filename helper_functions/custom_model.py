# Last updated February 22, 2024
# Version 0.1.4

import numpy as np
import pandas as pd
import time
from IPython.display import display
from typing import List, Literal, Any, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
    mean_squared_error,
    r2_score,
    make_scorer,
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    classification_report,
)
from imblearn.metrics import (
    classification_report_imbalanced,
    macro_averaged_mean_absolute_error,
)
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    f_regression,
    f_classif,
)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import shap
import warnings


def model_representation(
    pipe: Pipeline, type: str = "classif", labels: list = None
) -> None:
    """Representation of sklearn classifier or regressor.
    Type can be 'classif' or 'regression'."""
    if len(pipe) > 2:
        imputer = pipe[0]
        scaler = pipe[1]
        model = pipe[2]
        imputer_repr = pd.DataFrame(
            {imputer.strategy: imputer.statistics_},
            index=pipe.feature_names_in_,
        ).T
        print("IMPUTER")
        display(imputer_repr)
    else:
        scaler = pipe[0]
        model = pipe[1]
    scaler_repr = pd.DataFrame(
        {"Mean": scaler.mean_, "Scale": scaler.scale_},
        index=pipe.feature_names_in_,
    ).T
    print("SCALER")
    display(scaler_repr)
    if type == "classif":
        if labels is not None:
            idx = labels
        else:
            idx = pipe.classes_
        model_repr = pd.DataFrame(
            model.coef_, index=idx, columns=pipe.feature_names_in_
        )
        model_repr.index.name = "Class"
    elif type == "regression":
        model_repr = pd.DataFrame(model.coef_, index=pipe.feature_names_in_).T
    model_repr["Intercept"] = model.intercept_
    print("MODEL COEFFICIENTS")
    display(model_repr)
    return


def print_classification_results(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: List[str] = None,
    normalize: Literal["true", "pred", "all"] | None = None,
    show_cm: bool = True,
) -> None:
    """Function to print classification results: classification metrics for
    imbalanced data and confusion matrix."""
    print(
        "Classification metrics:"
        f" \n{classification_report_imbalanced(y_true, y_pred, target_names=labels, zero_division=0)}"
    )
    if show_cm:
        _, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            normalize=normalize,
            ax=ax,
            colorbar=False,
            cmap="coolwarm",
        )
        if labels is not None:
            ax.xaxis.set_ticklabels(labels)
            ax.yaxis.set_ticklabels(labels)
        plt.grid(False)
        plt.show()
    return


def print_regression_results(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_name: str,
    ylim: (float, float) = (None, None),
) -> Figure:
    """Function to print regression results: root mean square error,
    coefficient of determination, and scatterplots of predicted values against
    observed values, and of predicted values against residuals."""
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    if len(y_true) > 1e4:
        idx = y_true.sample(n=10000, random_state=42, ignore_index=False).index
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    residuals = y_true - y_pred
    scatter_kws = {"alpha": 0.4, "s": 10}
    line_kws = {"color": "rosybrown"}
    fig, axes = plt.subplots(1, 2, sharex=True)
    sns.regplot(
        x=y_pred,
        y=y_true,
        ax=axes[0],
        line_kws=line_kws,
        scatter_kws=scatter_kws,
        x_estimator=np.mean,
        x_ci="sd",
    )
    sns.regplot(
        x=y_pred,
        y=residuals,
        ax=axes[1],
        line_kws=line_kws,
        scatter_kws=scatter_kws,
        x_estimator=np.mean,
        x_ci="sd",
    )
    axes[0].set(xlabel=("Predicted " + y_name), ylabel=("Observed " + y_name))
    axes[1].set(xlabel=("Predicted " + y_name), ylabel="Residuals", ylim=ylim)
    print(f"RMSE = {round(RMSE, 3)}, R^2 = {round(r2, 3)}")
    return fig


def feature_selection_estimates(
    X: pd.DataFrame, y: pd.Series, type: str = "classif"
) -> pd.DataFrame:
    """Function to estimate Spearman's rank correlation coefficient,
    F-statistics and its p-values, and mutual information to accommodate feature
    selection. Type can be 'classif' or 'regression'."""
    measures = pd.DataFrame(index=X.columns)
    measures["Spearman's rho"] = X.corrwith(y, method="spearman")
    if type == "classif":
        f_stat, p_val = f_classif(X, y)
        mi = mutual_info_classif(X, y, random_state=0, discrete_features=False)
    elif type == "regression":
        f_stat, p_val = f_regression(X, y)
        mi = mutual_info_regression(
            X, y, random_state=0, discrete_features=False
        )
    measures["F-statistic"] = f_stat
    measures["p-value"] = p_val
    measures["Mutual Information"] = mi
    return measures


def select_k_best(
    X: pd.DataFrame, y: pd.Series, k: int, type: str = "classif"
) -> list:
    """Function to select top k features based on mutual information.
    Type can be 'classif' or 'regression'."""
    if type == "classif":
        mi = mutual_info_classif(X, y, random_state=0)
    elif type == "regression":
        mi = mutual_info_regression(X, y, random_state=0)
    return pd.Series(mi, index=X.columns).nlargest(n=k).index.to_list()


def vif_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the variance inflation factor for each feature in the dataframe."""
    vif_data = pd.DataFrame({"Feature": df.columns})
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(len(df.columns))
    ]
    return vif_data


def cv_score_binary_classifiers(
    clf_names: List[str],
    clf_models: List[object],
    fold: int | BaseCrossValidator,
    X: pd.DataFrame,
    y: pd.Series,
    beta: float = 2,
) -> pd.DataFrame:
    """Return mean scores on cross-validation of multiple binary classifiers."""
    recall = []
    precision = []
    auc = []
    average_precision = []
    accuracy = []
    balanced_accuracy = []
    f1 = []
    fbeta = []
    mcc = []
    time_taken = []
    scoring = {
        "recall": "recall",
        "precision": make_scorer(precision_score, zero_division=0.0),
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "fbeta": make_scorer(fbeta_score, beta=beta),
        "mcc": make_scorer(matthews_corrcoef),
    }
    for model in clf_models:
        start = time.time()
        cv_res = cross_validate(model, X, y, cv=fold, scoring=scoring)
        time_taken.append(time.time() - start)
        recall.append(cv_res["test_recall"].mean())
        precision.append(cv_res["test_precision"].mean())
        auc.append(cv_res["test_roc_auc"].mean())
        average_precision.append(cv_res["test_average_precision"].mean())
        accuracy.append(cv_res["test_accuracy"].mean())
        balanced_accuracy.append(cv_res["test_balanced_accuracy"].mean())
        f1.append(cv_res["test_f1"].mean())
        fbeta.append(cv_res["test_fbeta"].mean())
        mcc.append(cv_res["test_mcc"].mean())
    res_df = pd.DataFrame(
        {
            "AUC": auc,
            "Average Precision": average_precision,
            "Recall": recall,
            "Precision": precision,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "F1": f1,
            "F-beta": fbeta,
            "MCC": mcc,
            "Time": time_taken,
        },
        index=clf_names,
    )
    display(
        res_df.style.highlight_max(color="green", axis=0).format(precision=3)
    )
    return res_df


def score_binary_fitted_classifiers(
    clf_names: List[str],
    clf_models: List[object],
    X: pd.DataFrame,
    y: pd.Series,
    labels: List[str] = None,
    beta: float = 2,
) -> Figure:
    """Return scores of multiple binary fitted classifiers.
    Need to predict probabilities also."""
    recall = []
    precision = []
    auc_roc = []
    average_precision = []
    accuracy = []
    balanced_accuracy = []
    f1 = []
    fbeta = []
    mcc = []
    num_models = len(clf_models)
    if num_models > 3:
        num_rows, _ = divmod(num_models, 3)
        fig, axes = plt.subplots(num_rows + 1, 3)
    else:
        fig, axes = plt.subplots(1, num_models)
    if num_models > 1:
        axes = axes.flatten()
    for idx, model in enumerate(clf_models):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        recall.append(recall_score(y, y_pred))
        precision.append(precision_score(y, y_pred))
        auc_roc.append(roc_auc_score(y, y_proba[:, 1]))
        average_precision.append(average_precision_score(y, y_proba[:, 1]))
        accuracy.append(accuracy_score(y, y_pred))
        balanced_accuracy.append(balanced_accuracy_score(y, y_pred))
        f1.append(f1_score(y, y_pred))
        fbeta.append(fbeta_score(y, y_pred, beta=beta))
        mcc.append(matthews_corrcoef(y, y_pred))
        if num_models > 1:
            ConfusionMatrixDisplay.from_predictions(
                y,
                y_pred,
                display_labels=labels,
                ax=axes[idx],
                colorbar=False,
                values_format="d",
            )
            axes[idx].set_title(clf_names[idx])
            axes[idx].grid(False)
        else:
            ConfusionMatrixDisplay.from_predictions(
                y,
                y_pred,
                display_labels=labels,
                ax=axes,
                colorbar=False,
                values_format="d",
            )
            axes.set_title(clf_names[0])
            axes.grid(False)
    if num_models > 1:
        for ax in axes[num_models:]:
            ax.axis("off")
        fig.suptitle("Confusion Matrices", y=0.8)
    else:
        fig.suptitle("Confusion Matrix")
    plt.tight_layout()
    res_df = pd.DataFrame(
        {
            "AUC": auc_roc,
            "Average Precision": average_precision,
            "Recall": recall,
            "Precision": precision,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "F1": f1,
            "F-beta": fbeta,
            "MCC": mcc,
        },
        index=clf_names,
    )
    if num_models > 1:
        display(
            res_df.style.highlight_max(color="green", axis=0).format(
                precision=3
            )
        )
    else:
        display(res_df.style.format(precision=3))
    return res_df, fig


def reg_macro_averaged_mean_absolute_error(
    y_true: pd.Series, y_pred_continuous: np.ndarray
) -> float:
    """macro_averaged_mean_absolute_error extended to enable evaluation of
    regression based classification."""
    min = np.min(y_true)
    max = np.max(y_true)
    y_pred_discrete = pd.Series(y_pred_continuous.round())
    y_pred_discrete[y_pred_discrete < min] = min
    y_pred_discrete[y_pred_discrete > max] = max
    x = macro_averaged_mean_absolute_error(y_pred_discrete, y_true)
    return x


def plot_shap_values(
    model: Any,
    data: pd.DataFrame,
    processor: Pipeline = None,
    labels: List[str] = None,
    title: str = None,
    bar_height: float = "auto",
    max_display: int = 20,
) -> None:
    """Function to plot shap values of the model."""
    if processor:
        data = processor.transform(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = shap.TreeExplainer(model).shap_values(data)
    shap.summary_plot(
        shap_values,
        data,
        show=False,
        class_names=labels,
        class_inds="original",
        plot_size=bar_height,
        max_display=max_display,
    )
    plt.title(title)
    return


def draw_fitted_classifier_curves(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "Classifier Curves",
    size: Tuple[int, int] = None,
):
    y_proba = model.predict_proba(X)
    precision, recall, th = precision_recall_curve(y, y_proba[:, 1])
    fig, axes = plt.subplots(1, 3, figsize=size)
    RocCurveDisplay.from_predictions(
        y, y_proba[:, 1], plot_chance_level=True, ax=axes[0]
    )
    PrecisionRecallDisplay.from_predictions(y, y_proba[:, 1], ax=axes[1])
    sns.lineplot(x=th, y=precision[1:], label="Precision", ax=axes[2])
    sns.lineplot(x=th, y=recall[1:], label="Recall", ax=axes[2])
    axes[0].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Receiver Operating Characteristic Curve",
    )
    axes[1].set(
        xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve"
    )
    axes[2].set(
        xlabel="Threshold",
        xlim=(-0.01, 1.01),
        ylabel="Precision/Recall",
        ylim=(-0.01, 1.01),
        aspect="equal",
        title="Precision-Recall Curves vs. Threshold",
    )
    fig.suptitle(title)
    plt.tight_layout()
    return fig
