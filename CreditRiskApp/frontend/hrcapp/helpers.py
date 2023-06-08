from __future__ import annotations

import random

import matplotlib.pylab as plt
import palettable
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_object_dtype


def get_all_column_by_types(df, prefix=None, target_cols='TARGET', utilization=False):
    continuous_cols = []
    categorical_numerical_cols = []
    categorical_cols = []
    binary_cols = list(df.columns[(df.eq(0) | df.eq(1)).all()])

    if utilization:
        df['UTILIZATION_RATE'] = df['AMT_BALANCE'] / df[
            'AMT_CREDIT_LIMIT_ACTUAL'
        ].replace(0, 1)
        df['UTILIZATION_RATE'] = df['UTILIZATION_RATE'].fillna(0)

    for col in df.columns:
        if col not in binary_cols:
            if '_ID_' not in col and col not in [target_cols]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if len(df[col].unique()) <= 30 and 'mode' not in col.lower() and 'mean' not in col.lower():
                        if (prefix is None) or (
                            prefix is not None and col.startswith(prefix)
                        ):
                            categorical_numerical_cols.append(col)
                    else:
                        if (prefix is None) or (
                            prefix is not None and col.startswith(prefix)
                        ):
                            continuous_cols.append(col)
                else:
                    if (prefix is None) or (prefix is not None and col.startswith(prefix)):
                        if df[col].nunique() == 2 and ('Y' in df[col].unique() or 'Yes' in df[col].unique()):
                            df[[col]] = df[[col]].applymap(
                                lambda x: {'Y': 1, 'N': 0,
                                           'Yes': 1, 'No': 0}.get(x, x),
                            )
                            binary_cols.append(col)
                        else:
                            categorical_cols.append(col)

    return categorical_cols, categorical_numerical_cols, continuous_cols, binary_cols


def plot_kde(df, col, target=None):
    data = df.copy()

    if target:
        vals = data[target].value_counts()
    else:
        vals = [1]

    fig, _ = plt.subplots()
    sns.set_style('whitegrid')

    colors = palettable.cartocolors.qualitative.Vivid_10.hex_colors

    for i in range(0, len(vals)):
        if target:
            val = vals.keys()[i]
            data_to_plot = data[data[target] == val]
            label = f'{target} - {val}'
        else:
            data_to_plot = data
            label = None

        sns.kdeplot(
            data=data_to_plot,
            x=col,
            fill=True,
            common_norm=False,
            color=colors[i],
            alpha=0.5,
            linewidth=0,
            label=label,
        )

    if target:
        desc = f'Distribution of {col} and {target}'
    else:
        desc = f'Distribution of {col}'

    if target:
        plt.legend(loc='upper right')

    plt.title(desc, fontsize=13)
    st.pyplot(fig)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox('Add filters')

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        cols = [col for col in df.columns if col != 'SK_ID_CURR']
        to_filter_columns = st.multiselect('Filter dataframe on', cols)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f'Values for {column}',
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f'Values for {column}',
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(
                        map(pd.to_datetime, user_date_input),
                    )
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f'Substring or regex in {column}',
                )
                if user_text_input:
                    df = df[
                        df[column].astype(
                            str,
                        ).str.contains(user_text_input)
                    ]

    return df
