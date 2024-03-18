# Last updated March 8, 2024
# Version 0.1.0

from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from feature_engine.selection import (
    DropConstantFeatures,
    SmartCorrelatedSelection,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


class ApplicationsFeatureCreation(BaseEstimator, TransformerMixin):
    """Engineer features from applications table."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply necessary transformation to the data."""
        X["EXT_SOURCE_MIN"] = X[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].min(axis=1)
        X["EXT_SOURCE_MAX"] = X[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].max(axis=1)
        X["EXT_SOURCE_AVG"] = X[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].mean(axis=1)
        X["EXT_SOURCE_SUM"] = X[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].sum(axis=1)
        X["EXT_SOURCE_SUM"] = X[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].median(axis=1)
        X["EXT_SOURCE_PROD"] = X[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].prod(axis=1)
        X["CREDIT_ANNUITY_RATIO"] = X["AMT_CREDIT"] / X["AMT_ANNUITY"]
        X["ANNUITY_CREDIT_RATIO"] = X["AMT_ANNUITY"] / X["AMT_CREDIT"]
        X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]
        X["INCOME_CREDIT_RATIO"] = X["AMT_INCOME_TOTAL"] / X["AMT_CREDIT"]
        X["CREDIT_GOODS_RATIO"] = X["AMT_CREDIT"] / X["AMT_GOODS_PRICE"]
        X["CREDIT_GOODS_DIFF"] = X["AMT_CREDIT"] - X["AMT_GOODS_PRICE"]
        X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]
        X["INCOME_PER_CHILD"] = X["AMT_INCOME_TOTAL"] / (X["CNT_CHILDREN"] + 1)
        X["INCOME_PER_FAM_MEMBER"] = (
            X["AMT_INCOME_TOTAL"] / X["CNT_FAM_MEMBERS"]
        )
        X["YEARS_BIRTH"] = round(X["DAYS_BIRTH"] / 365).astype("int32")
        X["CAR_BIRTH_RATIO"] = X["OWN_CAR_AGE"] / X["YEARS_BIRTH"]
        X["EMPLOYED_BIRTH_RATIO"] = X["DAYS_EMPLOYED"] / X["DAYS_BIRTH"]
        return X

    def get_feature_names_out(self, input_features=None):
        return self._feature_names


class NNFeature(BaseEstimator, TransformerMixin):
    """Add feature of the mean TARGET value of N closest neighbors, based
    on EXT_SOURCE_X and CREDIT_ANNUITY_RATIO."""

    def __init__(self, n_neighbors=500):
        self.n_neighbors = n_neighbors
        self.nbrs = None

    def fit(self, X: pd.DataFrame, y=pd.Series):
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(
            X[[
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
                "CREDIT_ANNUITY_RATIO",
            ]]
        )
        self.y_train = y
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        nn_idx = self.nbrs.kneighbors(
            X[[
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
                "CREDIT_ANNUITY_RATIO",
            ]]
        )[1]
        X["NN_TARGET_MEAN"] = [self.y_train.iloc[idx].mean() for idx in nn_idx]
        return X

    def get_feature_names_out(self, input_features=None):
        return self._feature_names


def get_drop_multicollinear(
    X: pd.DataFrame, y: pd.Series, file_name: str, refit: bool = False
) -> List[str]:
    """Function to select quasi-constant and multicollinear features to drop
    based on their individual performance. Also can reload saved file."""
    if refit:
        dcf = DropConstantFeatures(missing_values="ignore", tol=0.995)
        dcf.fit(X)
        X = X.drop(columns=dcf.features_to_drop_)
        scs = SmartCorrelatedSelection(
            threshold=0.9,
            selection_method="model_performance",
            estimator=LogisticRegression(
                class_weight="balanced", random_state=42
            ),
        )
        scs.fit(X, y)
        drop_cols = dcf.features_to_drop_ + scs.features_to_drop_
        np.savetxt(
            f"./output/{file_name}.csv", drop_cols, fmt="%s", delimiter=","
        )
    else:
        drop_cols = np.loadtxt(f"./output/{file_name}.csv", dtype=str).tolist()
    print(
        "Number of quasi-constant and multicollinear aggregated features to"
        f" drop: {len(drop_cols)}"
    )
    return drop_cols


def get_drop_by_boruta(
    X: pd.DataFrame, y: pd.Series, file_name: str, refit: bool = False
) -> List[str]:
    """Function to select unimportant features to drop using Boruta. Also can
    reload saved file."""
    if refit:
        rf = RandomForestClassifier(
            class_weight="balanced", max_depth=5, random_state=42, n_jobs=-1
        )
        selector = BorutaPy(
            rf, n_estimators="auto", verbose=2, random_state=42
        )
        selector.fit(X.values, y.values)
        drop_cols = X.columns[~selector.support_].to_list()
        np.savetxt(
            f"./output/{file_name}.csv", drop_cols, fmt="%s", delimiter=","
        )
    else:
        drop_cols = np.loadtxt(f"./output/{file_name}.csv", dtype=str).tolist()
    print(
        "Number of unimportant aggregated features to drop using Boruta:"
        f" {len(drop_cols)}"
    )
    return drop_cols
