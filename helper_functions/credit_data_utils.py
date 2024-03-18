# Last updated February 19, 2024
# Version 0.1.0

from typing import Tuple, List
import polars as pl
from polars import col as c
import polars.selectors as cs
import pandas as pd
import numpy as np
import phik
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler


def reduce_memory_usage_pl(
    df: pl.DataFrame, verbose: bool = True
) -> pl.DataFrame:
    """Reduce memory usage by polars dataframe by changing its data types.
    Function taken from:
     https://www.kaggle.com/code/demche/polars-memory-usage-optimization.
    Original pandas version of this function:
    https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65"""
    if verbose:
        print(
            f"Size before memory reduction: {df.estimated_size('mb'):.2f} MB"
        )
        print(f"Initial data types {Counter(df.dtypes)}")
    Numeric_Int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    Numeric_Float_types = [pl.Float32, pl.Float64]
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in Numeric_Int_types:
            c_min = df[col].min() * 10  # prevent possible integer overflow
            c_max = df[col].max() * 10
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(df[col].cast(pl.Int8))
            elif (
                c_min > np.iinfo(np.int16).min
                and c_max < np.iinfo(np.int16).max
            ):
                df = df.with_columns(df[col].cast(pl.Int16))
            elif (
                c_min > np.iinfo(np.int32).min
                and c_max < np.iinfo(np.int32).max
            ):
                df = df.with_columns(df[col].cast(pl.Int32))
            elif (
                c_min > np.iinfo(np.int64).min
                and c_max < np.iinfo(np.int64).max
            ):
                df = df.with_columns(df[col].cast(pl.Int64))
        elif col_type in Numeric_Float_types:
            c_min = df[col].min()
            c_max = df[col].max()
            if (
                c_min > np.finfo(np.float32).min
                and c_max < np.finfo(np.float32).max
            ):
                df = df.with_columns(df[col].cast(pl.Float32))
            else:
                pass
        elif col_type == pl.String:
            df = df.with_columns(df[col].cast(pl.Categorical))
        else:
            pass
    if verbose:
        print(f"Size after memory reduction: {df.estimated_size('mb'):.2f} MB")
        print(f"Final data types {Counter(df.dtypes)}")
    return df


def initial_application_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    """Function to encode yes/no as 1/0, input None instead of XNA in
    application_train df. Also, remove correlated features about applicant's
    building."""
    print(f"Size before cleaning: {df.estimated_size('mb'):.2f} MB")
    print(f"Initial number of columns: {df.shape[1]}")
    df = df.with_columns(
        AMT_GOODS_PRICE=pl.when(c.AMT_GOODS_PRICE.is_null())
        .then(c.AMT_CREDIT)
        .otherwise(c.AMT_GOODS_PRICE),
        CODE_GENDER=pl.when(c.CODE_GENDER.str.contains("XNA"))
        .then(pl.lit(None))
        .otherwise(c.CODE_GENDER),
        FLAG_OWN_CAR=pl.when(c.FLAG_OWN_CAR.str.contains("Y"))
        .then(pl.lit(1))
        .when(c.FLAG_OWN_CAR.str.contains("N"))
        .then(pl.lit(0))
        .otherwise(pl.lit(None))
        .cast(pl.Int8),
        FLAG_OWN_REALTY=pl.when(c.FLAG_OWN_REALTY.str.contains("Y"))
        .then(pl.lit(1))
        .when(c.FLAG_OWN_REALTY.str.contains("N"))
        .then(pl.lit(0))
        .otherwise(pl.lit(None))
        .cast(pl.Int8),
        EMERGENCYSTATE_MODE=pl.when(c.EMERGENCYSTATE_MODE.str.contains("Y"))
        .then(pl.lit(1))
        .when(c.EMERGENCYSTATE_MODE.str.contains("N"))
        .then(pl.lit(0))
        .otherwise(pl.lit(None))
        .cast(pl.Int8),
        DAYS_EMPLOYED=pl.when(c.DAYS_EMPLOYED == 365243)
        .then(pl.lit(None))
        .otherwise(c.DAYS_EMPLOYED),
        OWN_CAR_AGE=c.OWN_CAR_AGE.round().cast(pl.Int64),
        NAME_FAMILY_STATUS=pl.when(c.NAME_FAMILY_STATUS == "Unknown")
        .then(pl.lit(None))
        .otherwise(c.NAME_FAMILY_STATUS),
        ORGANIZATION_TYPE=pl.when(c.ORGANIZATION_TYPE.str.contains("XNA"))
        .then(pl.lit(None))
        .otherwise(
            c.ORGANIZATION_TYPE.str.splitn(":", 2).struct.field("field_0")
        ),
        HOUR_APPR_PROCESS_START=c.HOUR_APPR_PROCESS_START.cast(pl.Utf8).cast(
            pl.Enum([str(num) for num in range(24)])
        ),
    )
    for col_name in df.columns:
        column = df[col_name]
        if (
            ("AMT_REQ_" in col_name)
            | ("CNT_" in col_name)
            | ("DAYS_" in col_name)
        ):
            df = df.with_columns(column.abs().cast(pl.Int64).alias(col_name))
    drop_building = df.select(
        cs.contains(["_MODE", "_MEDI"])
        & ~cs.contains([
            "HOUSETYPE",
            "TOTALAREA",
            "WALLSMATERIAL",
            "EMERGENCYSTATE_MODE",
            "FONDKAPREMONT",
        ])
    ).columns
    df = df.drop(columns=drop_building)
    print(f"Size after cleaning: {df.estimated_size('mb'):.2f} MB")
    print(f"Number of columns left: {df.shape[1]}")
    return df


def initial_bureau_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    """Function to clean bureau df."""
    df = df.with_columns(
        AMT_ANNUITY=c.AMT_ANNUITY.cast(pl.Float64),
        CREDIT_TYPE=pl.when(
            ~c.CREDIT_TYPE.str.contains(
                "Consumer|Credit|Mortgage|Car|Microloan"
            )
        )
        .then(pl.lit("Other"))
        .otherwise(c.CREDIT_TYPE),
        DAYS_ENDDATE_FACT=pl.when(c.DAYS_ENDDATE_FACT < -365 * 10)
        .then(c.DAYS_CREDIT_ENDDATE)
        .otherwise(c.DAYS_ENDDATE_FACT),
        DAYS_CREDIT_UPDATE=pl.when(c.DAYS_CREDIT_UPDATE < -365 * 10)
        .then(c.DAYS_ENDDATE_FACT.fill_null(0))
        .when(c.DAYS_CREDIT_UPDATE > 0)
        .then(-c.DAYS_CREDIT_UPDATE)
        .otherwise(c.DAYS_CREDIT_UPDATE),
        AMT_CREDIT_MAX_OVERDUE=pl.when(c.AMT_CREDIT_MAX_OVERDUE.is_null())
        .then(c.AMT_CREDIT_SUM_OVERDUE)
        .otherwise(c.AMT_CREDIT_MAX_OVERDUE),
    )
    df = df.with_columns(
        DAYS_CREDIT_ENDDATE=pl.when(c.DAYS_CREDIT_ENDDATE - c.DAYS_CREDIT < 0)
        .then(c.DAYS_ENDDATE_FACT)
        .otherwise(c.DAYS_CREDIT_ENDDATE)
    )
    for col_name in df.columns:
        column = df[col_name]
        if ("DAY_" in col_name) | ("CNT_" in col_name) | ("DAYS_" in col_name):
            df = df.with_columns(column.cast(pl.Int64).alias(col_name))
    return df


def initial_prev_application_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    """Function to encode yes/no as 1/0, input None instead of XNA in
    previous_application df, replace infinity days."""
    print(f"Size before cleaning: {df.estimated_size('mb'):.2f} MB")
    df = df.with_columns(
        HOUR_APPR_PROCESS_START=c.HOUR_APPR_PROCESS_START.cast(pl.Utf8).cast(
            pl.Enum([str(num) for num in range(24)])
        ),
        FLAG_LAST_APPL_PER_CONTRACT=pl.when(
            c.FLAG_LAST_APPL_PER_CONTRACT.str.contains("Y")
        )
        .then(pl.lit(1))
        .when(c.FLAG_LAST_APPL_PER_CONTRACT.str.contains("N"))
        .then(pl.lit(0))
        .otherwise(pl.lit(None)),
        SELLERPLACE_AREA=pl.when(c.SELLERPLACE_AREA == -1)
        .then(pl.lit(None))
        .otherwise(c.SELLERPLACE_AREA),
        PRODUCT_COMBINATION=c.PRODUCT_COMBINATION.str.splitn(
            ":", 2
        ).struct.field("field_0"),
    )
    for col_name in df.columns:
        column = df[col_name]
        if ("CODE_" in col_name) | ("NAME_" in col_name):
            df = df.with_columns(
                pl.when(column.str.contains("XNA"))
                .then(pl.lit(None))
                .otherwise(column)
                .alias(col_name)
            )
        elif "DAYS_" in col_name:
            df = df.with_columns(
                pl.when(column == 365243)
                .then(pl.lit(None))
                .otherwise(column)
                .cast(pl.Int64)
                .alias(col_name)
            )
        elif ("CNT_" in col_name) | ("FLAG_" in col_name):
            df = df.with_columns(column.cast(pl.Int64).alias(col_name))
    print(f"Size after cleaning: {df.estimated_size('mb'):.2f} MB")
    print(f"Number of columns left: {df.shape[1]}")
    return df


def initial_pos_cash_credit_card_installments_cleaning(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Function to force integers in CNT, NUM, DAYS columns, input None instead of XNA in
    pos_cash_balance, credit_card_balance and installments_payments dfs."""
    print(f"Size before cleaning: {df.estimated_size('mb'):.2f} MB")
    for col_name in df.columns:
        column = df[col_name]
        if "NAME_" in col_name:
            df = df.with_columns(
                pl.when(column.str.contains("XNA"))
                .then(pl.lit(None))
                .otherwise(column)
                .alias(col_name)
            )
        elif (
            ("CNT_" in col_name) | ("NUM_" in col_name) | ("DAYS_" in col_name)
        ):
            df = df.with_columns(column.cast(pl.Int64).alias(col_name))
    print(f"Size after cleaning: {df.estimated_size('mb'):.2f} MB")
    print(f"Number of columns left: {df.shape[1]}")
    return df


def get_column_types(
    df: pl.DataFrame,
) -> Tuple[list, list, list, list, list]:
    """Returns yes/no, categorical, rating, integer and float column types of
    credit tables."""
    yn_cols = []
    cat_cols = []
    rating_cols = []
    int_cols = []
    float_cols = []
    for col_name in df.columns:
        column = df[col_name]
        unique_count = column.n_unique()
        unique_values = column.unique().sort().to_list()
        if "RATING" in col_name:
            rating_cols.append(col_name)
        elif (
            (unique_count <= 3)
            & column.dtype.is_integer()
            & all(item in [0, 1, None] for item in unique_values)
        ):
            yn_cols.append(col_name)
        elif column.dtype.is_integer():
            int_cols.append(col_name)
        elif column.dtype.is_float():
            float_cols.append(col_name)
        elif (column.dtype == pl.Categorical) | (column.dtype == pl.Enum):
            cat_cols.append(col_name)
    assert (
        len(yn_cols)
        + len(rating_cols)
        + len(cat_cols)
        + len(int_cols)
        + len(float_cols)
    ) == df.shape[1], "not all columns assigned to types"
    return (yn_cols, cat_cols, rating_cols, int_cols, float_cols)


def get_balanced_pd_df(
    df: pd.DataFrame | pl.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Function to return randomly undersampled pandas dataframe so that it is
    balanced by indicated target column."""
    rus = RandomUnderSampler(random_state=42)
    if isinstance(df, pd.DataFrame):
        df_balanced, y = rus.fit_resample(
            df.drop(columns=[target]), df[target]
        )
    elif isinstance(df, pl.DataFrame):
        df_balanced, y = rus.fit_resample(
            df.drop([target]).to_pandas(), df.to_pandas()[target]
        )
    df_balanced = pd.concat([df_balanced, y], axis=1)
    print(f"Number of instances in balanced data: {df_balanced.shape[0]}")
    return df_balanced


def display_basic_info(df: pl.DataFrame) -> None:
    """Function to display size and duplicate information."""
    print("Number of instances:", df.shape[0])
    print("Number of duplicates:", df.is_duplicated().sum())
    return
