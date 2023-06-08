import copy
import math
import random

import matplotlib.pylab as plt
import numpy as np
import palettable
import pandas as pd
import scipy
import seaborn as sns
from category_encoders import BinaryEncoder
from keras.layers import Dense, Input
from keras.models import Model
from kneed import KneeLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import chi2_contingency, ttest_ind
from sklearn import metrics
from sklearn import pipeline as sklearn_pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import check_array

plt.rcParams["font.family"] = "Open Sans"
plt.rcParams["font.weight"] = "bold"


class AutomaticColumnTransformer(ColumnTransformer):
    def __init__(
        self,
        *,
        remainder="drop",
        selected_cols=[],
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
    ):
        self.transformers = []
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out
        self.selected_cols = selected_cols

    def fit_transform(self, X, y=None):
        (
            continuous_cols,
            categorical_cols,
            categorical_num_cols,
            binary_cols,
        ) = self._get_column_types(X[self.selected_cols])
        self.transformers = self.get_transformers(
            continuous_cols, categorical_cols, categorical_num_cols, binary_cols
        )

        return super().fit_transform(X, y)

    def _get_column_types(self, data):
        cat_cols, cat_num_cols, num_cols, binary_cols = get_all_column_by_types(
            data, target_cols="TARGET"
        )
        return num_cols, cat_cols, cat_num_cols, binary_cols

    def get_transformers(
        self, continuous_cols, categorical_cols, categorical_num_cols, binary_cols
    ):
        num_transformer = sklearn_pipeline.Pipeline(
            [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )

        cat_num_transformer = sklearn_pipeline.Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder()),
            ]
        )

        binary_transformer = sklearn_pipeline.Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent"))]
        )

        cat_transformer = sklearn_pipeline.Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, continuous_cols),
                ("binary", binary_transformer, binary_cols),
                ("cat", cat_transformer, categorical_cols),
                ("cat_num", cat_num_transformer, categorical_num_cols),
            ],
            remainder=self.remainder,
        )

        preprocessor.set_output(transform="pandas")

        transformers = preprocessor.transformers

        return transformers


class MissingMaskTransformer(TransformerMixin):
    def __init__(self, columns_to_impute):
        self.columns_to_impute = columns_to_impute

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        all_missing_mask = X[self.columns_to_impute].isnull().all(axis=1)
        X.loc[all_missing_mask, self.columns_to_impute] = -999
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def set_output(self, *, transform=None):
        if transform is None:
            return self

        if not hasattr(self, "_sklearn_output_config"):
            self._sklearn_output_config = {}

        self._sklearn_output_config["transform"] = transform
        return self


def rename_columns(X):
    original_column_names = []
    for col in X.columns:
        if "__" in col:
            new_col = col.split("__")
            original_column_names.append(new_col[1])
        else:
            original_column_names.append(col)

    X.columns = original_column_names
    return X


def plot_ipca_elbow_plot_and_get_n_components(X_train, threshold=0.85):
    """
    Given a training dataset X_train and a variance threshold, plots the elbow plot for incremental principal component analysis (IPCA)
    and returns the number of components to keep based on the given threshold.

    Parameters:
    X_train (array-like): The training dataset to perform IPCA on.
    threshold (float): A float value between 0 and 1 indicating the variance threshold to be used for determining
    the number of principal components to keep.

    Returns:
    int: The number of components to keep based on the variance threshold.

    Raises:
    ValueError: If the length of the input dataset X_train is less than or equal to 0.
    """
    X = X_train.copy()

    if scipy.sparse.issparse(X):
        X = X.toarray()

    if len(X) > 20000:
        batch_size = 10000
    elif len(X) > 10000:
        batch_size = 5000
    elif len(X) > 2000:
        batch_size = 1000
    elif len(X) > 300:
        batch_size = 100
    elif len(X) > 200:
        batch_size = 10
    else:
        batch_size = 1

    ipca = IncrementalPCA(n_components=None, batch_size=batch_size)

    for batch_X in np.array_split(X, len(X) // batch_size):
        ipca.partial_fit(batch_X)

    cumulative_var_ratio = np.cumsum(ipca.explained_variance_ratio_)

    n_components = np.argmax(cumulative_var_ratio >= threshold) + 1

    print("Number of components to keep:", n_components)
    plt.plot(cumulative_var_ratio)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.show()

    return n_components


def find_best_n_components(df, threshold=0.999):
    data = df.copy()
    # Check that X is a sparse array

    X = check_array(data, accept_sparse="csr")

    # Create a PCA object
    pca = PCA()

    # Fit the PCA object to the data
    pca.fit(X)

    # Compute the explained variance ratio for each component
    explained_variance_ratio = pca.explained_variance_ratio_

    # Compute the cumulative sum of explained variance ratios
    cum_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    # Find the index of the first component that exceeds the threshold
    n_components = np.argmax(cum_explained_variance_ratio >= threshold) + 1

    # Create a new PCA object with the best number of components
    best_pca = PCA(n_components=n_components)

    # Fit the new PCA object to the data
    best_pca.fit(X)

    # Return the number of components and the new PCA object
    return n_components, best_pca


def get_all_column_by_types(df, prefix=None, target_cols="TARGET", utilization=False):
    continuous_cols = []
    categorical_numerical_cols = []
    categorical_cols = []
    binary_cols = list(df.columns[(df.eq(0) | df.eq(1)).all()])

    if utilization:
        df["UTILIZATION_RATE"] = df["AMT_BALANCE"] / df[
            "AMT_CREDIT_LIMIT_ACTUAL"
        ].replace(0, 1)
        df["UTILIZATION_RATE"] = df["UTILIZATION_RATE"].fillna(0)

    for col in df.columns:
        if col not in binary_cols:
            if "_ID_" not in col and col not in [target_cols]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if (
                        len(df[col].unique()) <= 30
                        and "mode" not in col.lower()
                        and "mean" not in col.lower()
                    ):
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
                    if (prefix is None) or (
                        prefix is not None and col.startswith(prefix)
                    ):
                        if df[col].nunique() == 2 and (
                            "Y" in df[col].unique() or "Yes" in df[col].unique()
                        ):
                            df[[col]] = df[[col]].applymap(
                                lambda x: {"Y": 1, "N": 0, "Yes": 1, "No": 0}.get(x, x)
                            )
                            binary_cols.append(col)
                        else:
                            categorical_cols.append(col)

    return categorical_cols, categorical_numerical_cols, continuous_cols, binary_cols


def perform_cross_validation(pipelines, X_train, y_train, X_val, y_val, score="auc"):
    predictions = {}
    scores = {}

    for pipeline in pipelines:
        val_predictions = pipeline.predict(X_val)
        X_transformed = pipeline.named_steps["preprocessor"].transform(X_train)
        train_predictions = cross_val_predict(
            pipeline.named_steps["model"], X_transformed, y_train, cv=5
        )

        if score == "auc":
            fpr, tpr, _ = metrics.roc_curve(y_train, train_predictions, pos_label=1)
            train_score = metrics.auc(fpr, tpr)

            fpr, tpr, _ = metrics.roc_curve(y_val, val_predictions, pos_label=1)
            val_score = metrics.auc(fpr, tpr)

        model_name = type(pipeline.named_steps["model"]).__name__
        test_str = f"VALIDATION DATA - {score} for the {model_name} model is: {round(val_score, 3)}\n"
        cross_val_str = (
            f"TRAIN DATA (cross validated) {score} is {round(train_score, 3)}\n"
        )

        header_sep = "-" * int(
            round((max(len(test_str), len(cross_val_str)) - len(model_name) - 2) / 2)
        )
        print(f"{header_sep} {model_name} {header_sep}\n{cross_val_str}{test_str}")

        predictions[pipeline] = val_predictions
        scores[pipeline] = val_score

    return scores, predictions


def train_val_test_split(X_data, Y_data, target, validation=False):
    """
    Split the data into training, validation, and test sets while ensuring
    that the target variable is properly distributed among the splits.
    Parameters:
        X_data (pandas DataFrame): The features data to be split.
        Y_data (pandas DataFrame): The target variable data to be split.
        target (str): The name of the target variable column.
    Returns:
        tuple: A tuple of six elements containing the following:
            - X_train (pandas DataFrame): The training features data.
            - X_val (pandas DataFrame): The validation features data.
            - X_test (pandas DataFrame): The test features data.
            - y_train (pandas Series): The training target variable data.
            - y_val (pandas Series): The validation target variable data.
            - y_test (pandas Series): The test target variable data.
    """
    X = X_data.copy()
    X[target] = list(Y_data[target])

    categorical_features = list(X.select_dtypes(include=["object", "category"]).columns)
    categorical_features.append(target)

    X[categorical_features] = X[categorical_features].fillna("missing")

    if validation == False:
        X_train, X_test, y_train, y_test = train_test_split(
            X[[col for col in X.columns if col != target]],
            X[target],
            train_size=0.8,
            test_size=0.2,
            random_state=42,
            stratify=X[[target]],
        )

        for col in categorical_features:
            X[col] = X[col].apply(lambda x: None if x == "missing" else x)

        return X_train, X_test, y_train, y_test

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1, stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_pipelines(models, preprocess, pca):
    pipelines = []
    for model in models:
        pipeline = sklearn_pipeline.Pipeline(
            [
                ("preprocessor", preprocess),
                ("PCA", pca),
                ("model", model),
            ]
        )
        pipelines.append(pipeline)

    return pipelines


def plot_mi_scores(scores):
    """
    Plot a horizontal bar chart of mutual information scores.
    Parameters
    ----------
    scores : pandas.Series
        A series containing the mutual information scores to be plotted.
    Returns
    -------
    None
        The function displays the plot but does not return anything.
    """
    plt.figure(dpi=100, figsize=(15, 12))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(
        width, scores, color=palettable.cartocolors.qualitative.Vivid_10.hex_colors[3]
    )
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()


def make_mi_scores(X, y):
    """
    Calculate the mutual information scores between features and target variable.
    Parameters
    ----------
    X : pandas.DataFrame
        The input data containing the features to be evaluated.
    y : pandas.Series
        The target variable used to calculate the mutual information scores.
    Returns
    -------
    pandas.Series
        A series containing the mutual information scores for each feature in X.
        The series is sorted in descending order based on the scores.
    """
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=0
    )
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def get_and_plot_mi_scores(df, target_df, target, n=70):
    """
    This function calculates and plots mutual information (MI) scores for features in a dataframe with respect to a target variable.
    Args:
    df: pandas DataFrame containing the features.
    target_df: pandas DataFrame containing the target variable.
    target: name of the target variable.
    Returns:
    A dictionary of MI scores for each feature.
    A plot of the MI scores.
    """
    X = df[[col for col in df.columns if col != target]].copy()
    X[target] = target_df[target]
    X = X.dropna()

    y = X[target]
    X = X[[col for col in X.columns if col != target]].copy()

    mi_scores = make_mi_scores(X, y)
    plot_mi_scores(mi_scores[:n])
    return mi_scores


def get_grouped_dataframe_by_ID(data, ID, prefix, other_prefixes):
    df = data.copy()

    df_agg = df.groupby(ID).agg(["mean"])
    df_agg.columns = [
        "_".join(x) if type(x) == tuple else x for x in df_agg.columns.ravel()
    ]
    df_agg.columns = [
        f"{prefix}_{col}"
        if col != ID and not any(col.startswith(val) for val in other_prefixes)
        else col
        for col in df_agg.columns
    ]

    ID_cols = [
        col for col in df_agg.columns if "_ID_" in col and not col.startswith("SK_ID")
    ]
    df_agg = df_agg.drop(ID_cols, axis=1)

    return df_agg


def perform_multiple_hypothesis_tests_categorical(df, cols, target):
    dataset = df.copy()
    results_df = pd.DataFrame(columns=["Column", "Test Statistic", "p-value"])

    for col in cols:
        data = dataset.copy()
        data.dropna(subset=[col], inplace=True)

        # Create a contingency table of the data
        contingency_table = pd.crosstab(data[target], data[col])

        # Perform the chi-squared test
        stat, p_value, _, _ = chi2_contingency(contingency_table)

        if p_value < 0.05:
            nh = "REJECTED"
        else:
            nh = "could not REJECTED"

        # Append the results to the DataFrame
        results_df = results_df.append(
            {
                "Column": col,
                "Test Statistic": stat,
                "p-value": p_value,
                "Null Hypothesis": nh,
            },
            ignore_index=True,
        )

    return results_df


def perform_multiple_hypothesis_tests_continuous(df, cols, target):
    dataset = df.copy()
    results_df = pd.DataFrame(
        columns=[
            "Column",
            "Mean for Group 1",
            "Mean for Group 2",
            "Test Statistic",
            "p-value",
        ]
    )

    for col in cols:
        data = dataset.copy()
        data.dropna(subset=[col], inplace=True)

        group1 = data[data[target] == list(data[target].unique())[0]]
        group2 = data[data[target] == list(data[target].unique())[1]]

        sample_size = len(group1) if len(group2) > len(group1) else len(group2)

        target_val_1 = group1.sample(n=sample_size, random_state=1)[col]
        target_val_2 = group2.sample(n=sample_size, random_state=1)[col]

        mean_val1 = np.mean(target_val_1)
        mean_val2 = np.mean(target_val_2)

        stat, p_value = ttest_ind(target_val_1, target_val_2)

        if p_value < 0.05:
            nh = "REJECTED"
        else:
            nh = "could not REJECTED"

        results_df = results_df.append(
            {
                "Column": col,
                "Mean for Group 1": round(mean_val1, 2),
                "Mean for Group 2": round(mean_val2, 2),
                "Test Statistic": stat,
                "p-value": p_value,
                "Null Hypothesis": nh,
            },
            ignore_index=True,
        )

    return results_df


def plot_two_cols_kde(df, col, target=None, without_outliers=None):
    data = df.copy()

    if without_outliers:
        # Calculate the 25th and 75th percentiles (Q1 and Q3)
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)

        # Calculate the interquartile range (IQR)
        iqr = q3 - q1

        # Calculate the upper limit for outliers
        upper_limit = q3 + 1.5 * iqr

        # Remove the top 5% outliers
        data = data[data[col] <= upper_limit]

    if target:
        vals = data[target].value_counts()

    plt.figure(figsize=(15, 8))
    colors = palettable.cartocolors.qualitative.Vivid_10.hex_colors
    random.shuffle(colors)

    for i in range(0, len(vals)):
        val = vals.keys()[i]
        sns.kdeplot(
            data=data[data[target] == val],
            x=col,
            fill=True,
            common_norm=False,
            color=colors[i],
            alpha=0.5,
            linewidth=0,
            label=f"{target} - {val}",
        )

    outliers_removal = "without outliers" if without_outliers else ""

    plt.legend(loc="upper right")
    plt.title(f"Distribution of {col} and {target} {outliers_removal}", fontsize=13)
    plt.show()


def plot_top6_hypothesis_test_cols(df, top6_cols, target):
    data = df.copy()

    vals = list(data[target].unique())
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))

    colors = palettable.mycarta.Cube1_12.hex_colors
    random.shuffle(colors)

    k = 0
    for i in range(0, 2):
        for j in range(0, 3):
            ax = axes[i][j]
            val = top6_cols[k]

            sns.kdeplot(
                data=data[data[target] == vals[0]],
                x=val,
                fill=True,
                common_norm=False,
                color=colors[k],
                alpha=0.5,
                linewidth=0,
                label=f"{target} - {val}",
                ax=ax,
            )
            sns.kdeplot(
                data=data[data[target] == vals[1]],
                x=val,
                fill=True,
                common_norm=False,
                color=colors[k + 1],
                alpha=0.5,
                linewidth=0,
                label=f"{target} - {val}",
                ax=ax,
            )

            k += 1

            ax.set_title(f"{target} - {val}")
            ax.legend(loc="upper right")
            ax.set_xlabel(val)
            ax.set_ylabel("Density")

    fig.suptitle(
        f"Distribution of Top 10 colums that have signifance difference with different {target} values",
        fontsize=20,
    )
    plt.tight_layout()
    plt.show()


def plot_multiple_plots_two_categorical(
    cat1, cat2, cat1_label, cat2_label, df, hide_small_cats=False, num_to_hide=40
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # plot the pie chart on the left
    colors = palettable.mycarta.Cube1_9.hex_colors
    contract_types = df[cat1].unique()

    ax = axes[0][0]
    ax.add_artist(plt.Circle((0, 0), 0.8 - 0.3 / 2, color="white"))
    ax.text(
        0,
        0,
        f"{len(df):.0f}\nLoans",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    value_counts = df[cat1].value_counts()

    labels = []
    k = 0
    for key, val in dict(df[cat1].value_counts()).items():
        if len(value_counts) - k <= num_to_hide and hide_small_cats == True:
            labels.append("")
        else:
            labels.append(f"{key}\n{round(val/len(df)*100, 4)}%")
        k += 1

    value_counts.plot(
        kind="pie",
        labels=labels,
        colors=colors[: len(contract_types)],
        labeldistance=1.2,
        ax=ax,
        wedgeprops=dict(width=0.3, edgecolor="white"),
        startangle=120,
    )

    df_counts1 = value_counts.reset_index()
    df_counts1.columns = [cat1, "count"]

    if len(labels) >= 3:
        ax.texts[3].set_position(
            (ax.texts[3].get_position()[0] + 0.15, ax.texts[3].get_position()[1] + 0.1)
        )
        ax.texts[4].set_position(
            (ax.texts[4].get_position()[0] + 0.05, ax.texts[4].get_position()[1] + 0.0)
        )
        ax.texts[5].set_position(
            (
                ax.texts[5].get_position()[0] - 0.05,
                ax.texts[5].get_position()[1] - 0.155,
            )
        )

    ax.set_axis_off()

    # plot the count plot on the right
    sns.countplot(x=cat1, hue=cat2, data=df, ax=axes[0][1], palette=colors)
    axes[0][1].set_title(f"Loan Counts by {cat1_label} and {cat2_label}", fontsize=13)
    axes[0][1].legend(title=cat2_label, loc="upper right")
    axes[0][1].set_xticklabels(axes[0][1].get_xticklabels(), rotation=25, fontsize=8)

    axes[1][0].set_title(
        f"{cat1_label} values where {cat2_label} is {list(df[cat2].unique())[0]}",
        fontsize=13,
    )
    pie_values = df[df[cat2] == list(df[cat2].unique())[0]][cat1].value_counts()
    df_counts2 = pie_values.reset_index()
    df_counts2.columns = [
        f"{cat1} where {cat2} == {list(df[cat2].unique())[0]}",
        "count",
    ]

    labels = []
    k = 0
    for key, val in dict(pie_values).items():
        if len(pie_values) - k <= (num_to_hide + 5) and hide_small_cats == True:
            labels.append("")
        else:
            labels.append(f"{key}\n{round(val/len(df)*100, 4)}%")
        k += 1

    pie_values.plot(
        kind="pie", ax=axes[1][0], colors=colors[: len(contract_types)], labels=labels
    )
    axes[1][0].set_axis_off()

    axes[1][1].set_title(
        f"{cat1_label} values where {cat2_label} is {list(df[cat2].unique())[1]}",
        fontsize=13,
    )
    pie_values = df[df[cat2] == list(df[cat2].unique())[1]][cat1].value_counts()
    df_counts3 = pie_values.reset_index()
    df_counts3.columns = [
        f"{cat1} where {cat2} == {list(df[cat2].unique())[1]}",
        "count",
    ]

    labels = []
    k = 0
    for key, val in dict(pie_values).items():
        if len(pie_values) - k <= (num_to_hide + 5) and hide_small_cats == True:
            labels.append("")
        else:
            labels.append(f"{key}\n{round(val/len(df)*100, 4)}%")
        k += 1

    pie_values.plot(
        kind="pie", ax=axes[1][1], colors=colors[: len(contract_types)], labels=labels
    )
    axes[1][1].set_axis_off()

    # adjust the spacing between subplots
    plt.subplots_adjust(wspace=0)

    # add a title for the entire plot
    suptitle = fig.suptitle(
        f"Loan Default Analysis by {cat1_label}", fontsize=16, fontweight="bold"
    )
    suptitle.set_y(1)

    fig.tight_layout(h_pad=2.0)
    # show the plot
    plt.show()

    if hide_small_cats:
        print(f"Overall distrubution of {cat1}")
        display(df_counts1.head(5))

        print(f"Overall distrubution of {cat1}")
        display(df_counts2.head(5))

        print(f"Overall distrubution of {cat1}")
        display(df_counts3.head(5))


def plot_numerical_vals_box_plots_and_return_outliers(data, continuous_cols, df_name):
    df = data[continuous_cols].copy()

    n_cols = 6
    n_rows = math.ceil(len(continuous_cols) / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(20, 25),
        gridspec_kw={"wspace": 0.4, "hspace": 0.4},
    )

    for i, column in enumerate(continuous_cols):
        row = i // n_cols
        col = i % n_cols

        df.boxplot(column=column, ax=axes[row, col])
        axes[row, col].set_title(column)

    for ax in axes.flatten():
        if not ax.get_title():
            ax.set_visible(False)

    fig.suptitle(
        f"Distribution of numerical values in {df_name} dataframe", fontsize=16, y=1.03
    )
    fig.subplots_adjust(top=0.99)
    plt.show()


def drop_duplicates_from_df(data):
    df = data.copy()
    old_len = len(df)
    df.drop_duplicates(inplace=True)
    if old_len == len(df):
        print("No duplicates found.")
    else:
        print(
            f"Size of dataframe before removing duplicates - {old_len}, and after removing duplicates - {len(data)}"
        )
    return df


def plot_kde(df, col, target=None, without_outliers=None, outliers_threshold=0.25):
    data = df.copy()

    if without_outliers:
        q1 = data[col].quantile(outliers_threshold)
        q3 = data[col].quantile(1 - outliers_threshold)

        iqr = q3 - q1

        upper_limit = q3 + ((1 - outliers_threshold) * 2) * iqr

        # Remove the top 5% outliers
        data = data[data[col] <= upper_limit]

    if target:
        vals = data[target].value_counts()
    else:
        vals = [1]

    plt.figure(figsize=(15, 8))
    colors = palettable.cartocolors.qualitative.Vivid_10.hex_colors
    random.shuffle(colors)

    for i in range(0, len(vals)):
        if target:
            val = vals.keys()[i]
            data_to_plot = data[data[target] == val]
            label = f"{target} - {val}"
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

    outliers_removal = "without outliers" if without_outliers else ""

    if target:
        desc = f"Distribution of {col} and {target} {outliers_removal}"
    else:
        desc = f"Distribution of {col} {outliers_removal}"

    if target:
        plt.legend(loc="upper right")

    plt.title(desc, fontsize=13)
    plt.show()


def plot_scatter_for_two_cols(data, col1, col2):
    plt.figure(figsize=(15, 8))
    colors = palettable.cartocolors.qualitative.Vivid_10.hex_colors
    random.shuffle(colors)

    amt_balance = data["AMT_BALANCE"]
    amt_payment_total = data["AMT_PAYMENT_TOTAL_CURRENT"]

    # Create a scatter plot
    plt.scatter(amt_balance, amt_payment_total, alpha=0.5, color=colors[0])

    # Adding labels and title
    plt.xlabel("Balance Amount")
    plt.ylabel("Total Payment")
    plt.title("Relationship between Balance Amount and Total Payment")

    # Display the scatter plot
    plt.show()


def plot_grouped_by_col_data(data, col, descriptions, groupedby="Contract Status"):
    display(
        descriptions[
            (descriptions.Row == col)
            & (descriptions.Table == "credit_card_balance.csv")
        ]
        .iloc[0]
        .Description
    )
    display(data[col])
    colors = palettable.cartocolors.qualitative.Vivid_10.hex_colors
    random.shuffle(colors)
    data[col]["mean"].sort_values().plot(kind="bar", figsize=(15, 8), color=colors[0])
    plt.xlabel(groupedby)
    plt.ylabel(col)
    plt.title(f"{col} by {groupedby}")
    plt.show()


def plot_missing_vals(df, missing_cols):
    if len(missing_cols) < 10:
        for col in missing_cols:
            print(
                f"{col} column missing values percentage - {round(df[col].isnull().sum() / len(df) * 100, 8)}%"
            )

    plt.figure(figsize=(15, 8))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
    plt.title("Missing Values")
    plt.show()


def plot_dict_barplot(vals, keys, values, title):
    display(vals)
    colors = palettable.cartocolors.qualitative.Vivid_10.hex_colors
    random.shuffle(colors)

    plt.figure(figsize=(15, 8))
    vals.plot.bar(color=colors[0])

    plt.xlabel(keys)
    plt.ylabel(values)
    plt.title(title)

    plt.xticks(rotation=90)
    plt.show()


def plot_feature_correlation_map(df, cmap="coolwarm"):
    corr_df = df.corr()
    plt.figure(figsize=(13, 9))
    sns.heatmap(corr_df, mask=np.triu(corr_df), annot=True, fmt=".0%", cmap=cmap)
    plt.title("Feature Correlation Heatmap")
    plt.show()


def plot_anomaly_fraud_scores_with_col(df, col, descriptions, fraud_col="fraud_scores"):
    description = (
        descriptions[
            (descriptions.Row == col)
            & (descriptions.Table == "credit_card_balance.csv")
        ]
        .iloc[0]
        .Description
    )
    print(f"{description} and fraud_scores relationship")
    plt.figure(figsize=(15, 8))
    plt.scatter(range(len(df)), df[col], c=list(df[fraud_col]), cmap="coolwarm")
    plt.colorbar(label="Anomaly Score")
    plt.xlabel("Data Point Index")
    plt.ylabel(col)
    plt.title(f"Anomaly Scores for {col}")
    plt.show()


def get_anomaly_detection_pipeline(
    continuous_cols,
    categorical_num_cols,
    categorical_cols,
    binary_cols,
    cols_impute_by_zero,
):
    zero_cols_transformer = sklearn_pipeline.Pipeline(
        [("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    )

    num_transformer = sklearn_pipeline.Pipeline(
        [("imputer", SimpleImputer(strategy="mean"))]
    )

    binary_transformer = sklearn_pipeline.Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    cat_num_transformer = sklearn_pipeline.Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder()),
        ]
    )

    cat_transformer = sklearn_pipeline.Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", BinaryEncoder()),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("zero_num", zero_cols_transformer, cols_impute_by_zero),
            ("num", num_transformer, continuous_cols),
            ("binary", binary_transformer, binary_cols),
            ("cat_num", cat_num_transformer, categorical_num_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="passthrough",
    )

    column_transformer.set_output(transform="pandas")

    return sklearn_pipeline.Pipeline(
        [("preprocessor", column_transformer), ("isolation_forest", IsolationForest())]
    )


def plot_isolation_forest_feature_importance(df, pipeline):
    feature_importance = np.mean(
        [
            tree.feature_importances_
            for tree in pipeline.named_steps["isolation_forest"].estimators_
        ],
        axis=0,
    )
    feature_importances_with_cols = pd.Series(feature_importance, index=df.columns)
    feature_importances_with_cols = feature_importances_with_cols.sort_values()

    # Normalize feature importance values to [0, 1]
    normalized_values = (
        feature_importances_with_cols - feature_importances_with_cols.min()
    ) / (feature_importances_with_cols.max() - feature_importances_with_cols.min())

    # Define colormap
    colormap = plt.cm.get_cmap("viridis_r")  # Choose a colormap of your preference

    # Plotting
    plt.figure(dpi=100, figsize=(15, 8))
    colors = colormap(normalized_values)
    feature_importances_with_cols.plot.bar(color=colors, width=0.75)
    plt.xlabel("Features")
    plt.ylabel("Feature Importance")
    plt.title("Feature Importance in Isolation Forest")

    # Create a colorbar legend
    norm = Normalize(
        vmin=feature_importances_with_cols.min(),
        vmax=feature_importances_with_cols.max(),
    )
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Feature Importance")

    plt.show()


def plot_anomaly_scores(df):
    plt.figure(dpi=100, figsize=(15, 8))

    # Plot histogram
    plt.subplot(1, 2, 1)
    plt.hist(df["anomaly_score"], bins="auto", color="#22544f")
    plt.title("Anomaly/Fraud Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")

    # Plot pie chart
    plt.subplot(1, 2, 2)
    fraud_counts = df["is_fraud"].value_counts()
    labels = ["Not Fraud", "Fraud"]
    colors = ["#3498db", "#e74c3c"]
    plt.pie(fraud_counts, labels=labels, colors=colors, autopct="%1.1f%%")
    plt.title("Fraud Distribution")

    plt.tight_layout()
    plt.show()


def select_best_categorical_features(dataframe, categorical_features, target):
    # Compute the chi-square statistic and p-value for each categorical feature
    chi2_scores = []
    p_values = []

    for feature in categorical_features:
        contingency_table = pd.crosstab(dataframe[feature], dataframe[target])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        chi2_scores.append(chi2)
        p_values.append(p_value)

    # Rank the categorical features based on their p-values
    ranked_features = sorted(
        zip(categorical_features, p_values, chi2_scores), key=lambda x: x[1]
    )

    # Select the best categorical features based on a significance threshold (e.g., 0.05)
    best_features = [
        feature for feature, p_value, _ in ranked_features if p_value < 0.05
    ]

    return best_features


def create_cols_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            pairs.append((lst[i], lst[j]))
    return pairs


def plot_K_means_elbow(X):
    wcss = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        wcss.append(km.inertia_)

    kl = KneeLocator(K, wcss, curve="convex", direction="decreasing")
    print("The optimal number of clusters is:", kl.elbow)

    n_clusters = kl.elbow

    plt.plot(K, wcss, "bx-")
    plt.xlabel("Number of centroids")
    plt.ylabel("WCSS")
    plt.title("Elbow Method For Optimal k")
    plt.show()

    return n_clusters


def find_missing_data_patterns(df, missing_num_cols):
    missing_patterns = []
    df = df.sample(1000, random_state=42).copy()

    for col in missing_num_cols:
        missing_pattern = [col]
        missing_rows = df[df[col].isnull()]

        if len(missing_rows) > 100:
            for row in missing_rows.index:
                missing_cols = df.loc[row][df.loc[row].isnull()].index.tolist()
                selected_rows = df[df[missing_cols].isna().all(axis=1)]

                if len(missing_cols) > 1:
                    if len(selected_rows) == len(missing_rows):
                        missing_pattern.extend(list(set(missing_cols)))

            if len(missing_pattern) > 1:
                missing_patterns.append(list(set(missing_pattern)))

    return missing_patterns


def create_new_application_features(data):
    df = data.copy()
    df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    return df


known_replace_by_zero_cols = [
    "CNT_FAM_MEMBERS",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU",
    "SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
]


def get_application_cleaner(df, missing_patterns, fill_by_None, replace_by_zero):

    fill_Unaccompanied = [
        col for col in df.columns if "Unaccompanied" in list(df[col].unique())
    ]
    replace_with_OTHER = [
        col for col in df.columns if "OTHER" in list(df[col].unique())
    ]
    replace_by_No = [col for col in df.columns if "No" in list(df[col].unique())]
    columns_to_impute = missing_patterns

    mean_imputer = SimpleImputer(strategy="mean")
    constant_imputer = SimpleImputer(strategy="constant", fill_value=-999)

    final_preprocessor = sklearn_pipeline.Pipeline([])
    i = 0

    already_imputed_cols = []
    for columns_to_impute in missing_patterns:
        missing_pipeline = sklearn_pipeline.Pipeline(
            [
                ("missing_mask", MissingMaskTransformer(columns_to_impute)),
                ("imputer", constant_imputer),
                ("imputer_none", mean_imputer),
                (
                    "columns_renamer",
                    preprocessing.FunctionTransformer(rename_columns),
                ),
            ]
        )

        pattern_imputer = ColumnTransformer(
            transformers=[("pattern_impute", missing_pipeline, columns_to_impute)],
            remainder="passthrough",
        )

        pattern_imputer.set_output(transform="pandas")

        pipeline = sklearn_pipeline.Pipeline(
            [
                ("pattern_imputer", pattern_imputer),
                (
                    "columns_renamed_1",
                    preprocessing.FunctionTransformer(rename_columns),
                ),
            ]
        )
        final_preprocessor.steps.append((f"missing_cols_cleaner_{i}", pipeline))
        i += 1
        already_imputed_cols.extend(columns_to_impute)

    already_imputed_cols = list(set(already_imputed_cols))
    (
        categorical_cols,
        categorical_numerical_cols,
        continuous_cols,
        binary_cols,
    ) = get_all_column_by_types(df)
    replace_by_zero = list(
        set([col for col in replace_by_zero if col not in already_imputed_cols])
    )

    all_cat_cols = list(
        set(categorical_cols + categorical_numerical_cols + binary_cols)
    )
    continuous_cols = list(
        set(
            [
                col
                for col in continuous_cols
                if col not in fill_Unaccompanied
                and col not in fill_by_None
                and col not in replace_with_OTHER
                and col not in replace_by_No
                and col not in replace_by_zero
                and col not in continuous_cols
                and col not in all_cat_cols
                and col not in already_imputed_cols
            ]
        )
    )
    all_cat_cols_for_impute = list(
        set(
            [
                col
                for col in all_cat_cols
                if col not in fill_Unaccompanied
                and col not in fill_by_None
                and col not in replace_with_OTHER
                and col not in replace_by_No
                and col not in replace_by_zero
                and col not in continuous_cols
                and col not in already_imputed_cols
            ]
        )
    )

    constant_imputer = ColumnTransformer(
        transformers=[
            (
                "imputer_Unaccompanied",
                SimpleImputer(strategy="constant", fill_value="Unaccompanied"),
                [
                    col
                    for col in list(set(fill_Unaccompanied))
                    if col not in already_imputed_cols
                ],
            ),
            (
                "imputer_None",
                SimpleImputer(strategy="constant", fill_value="None"),
                [
                    col
                    for col in list(set(fill_by_None))
                    if col not in already_imputed_cols
                ],
            ),
            (
                "imputer_OTHER",
                SimpleImputer(strategy="constant", fill_value="OTHER"),
                [
                    col
                    for col in list(set(replace_with_OTHER))
                    if col not in already_imputed_cols
                ],
            ),
            (
                "imputer_No",
                SimpleImputer(strategy="constant", fill_value="No"),
                [
                    col
                    for col in list(set(replace_by_No))
                    if col not in already_imputed_cols
                ],
            ),
            (
                "imputer_Zero",
                SimpleImputer(strategy="constant", fill_value=0),
                [
                    col
                    for col in list(set(replace_by_zero))
                    if col not in already_imputed_cols
                ],
            ),
            ("imputer_Mean", SimpleImputer(strategy="mean"), continuous_cols),
            (
                "imputer_Mode",
                SimpleImputer(strategy="most_frequent"),
                all_cat_cols_for_impute,
            ),
        ],
        remainder="passthrough",
    )
    constant_imputer.set_output(transform="pandas")

    pipeline = sklearn_pipeline.Pipeline(
        [
            ("pattern_imputer", constant_imputer),
            ("columns_renamed_2", preprocessing.FunctionTransformer(rename_columns)),
        ]
    )

    final_preprocessor.steps.append((f"cleaner_imputer", pipeline))

    pipeline = sklearn_pipeline.Pipeline(
        [
            (
                "feature_engineering",
                preprocessing.FunctionTransformer(create_new_application_features),
            )
        ]
    )

    final_preprocessor.steps.append(("new_feature_creation", pipeline))

    return final_preprocessor


def get_final_application_preprocessor_pipeline(
    cleaner, selected_cols, others_drop="drop"
):
    final_preprocessor = sklearn_pipeline.Pipeline([])
    final_preprocessor.steps.append((f"cleaner", cleaner))

    selected_cols_with_new = [
        "CREDIT_INCOME_PERCENT",
        "ANNUITY_INCOME_PERCENT",
        "CREDIT_TERM",
        "DAYS_EMPLOYED_PERCENT",
    ] + selected_cols

    preprocessor = AutomaticColumnTransformer(
        remainder=others_drop, selected_cols=selected_cols_with_new
    )
    preprocessor.set_output(transform="pandas")
    final_preprocessor.steps.append((f"final_preprocessor", preprocessor))

    final_preprocessor.steps.append(
        ("prefixes_removal", preprocessing.FunctionTransformer(rename_columns))
    )

    return final_preprocessor


def perform_keras_encode_data(df, epochs=100, batch_size=1000):
    data = df.values

    input_dim = data.shape[1]
    encoding_dim = 3

    input_data = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(input_data)
    decoded = Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = Model(input_data, decoded)

    encoder = Model(input_data, encoded)

    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=0)
    encoder.compile()
    
    return encoder, encoder.predict(data)


def plot_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=clusters)
    ax.set_xlabel("Encoded feature 1")
    ax.set_ylabel("Encoded feature 2")
    ax.set_zlabel("Encoded feature 3")
    plt.title("Clustering Results")
    plt.show()

    return clusters, kmeans


def plot_kde_for_clusters_by_col(data, col):
    CLUSTER_MAPPING = {1: "Low Risk", 0: "Medium Risk", 2: "High Risk"}

    df = data.copy()
    plt.figure(dpi=100, figsize=(15, 8))

    df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    clusters = list(df["cluster"].unique())
    clusters.sort()

    for cluster in clusters:
        sns.kdeplot(
            df[df["cluster"] == cluster][col],
            fill=True,
            common_norm=False,
            alpha=0.5,
            linewidth=0,
            label=f"cluster = {CLUSTER_MAPPING[cluster]}",
        )

    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.title(f"{col} values by cluster")
    plt.show()


def get_missing_cols_from_application_cleaner(final_preprocessor, final_features):
    cleaner = final_preprocessor.named_steps["cleaner"]

    m_pats = []
    for step in cleaner.named_steps:
        try:
            for transformer in (
                cleaner.named_steps[step].named_steps["pattern_imputer"].transformers
            ):
                if transformer[0] == "pattern_impute":
                    used_cols = set(transformer[2])
                    final_used_cols = used_cols.intersection(set(final_features))
                    if len(final_used_cols) > 1:
                        m_pats.append(list(final_used_cols))
        except:
            pass

    return m_pats


def get_too_correlated_cols(df, continuous_cols, mi_scores):
    data = df[continuous_cols + ["TARGET"]].copy()
    corr_matrix = data[continuous_cols].corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    )

    threshold = 0.95
    corr_pairs = {}
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][
            upper_triangle[column] > threshold
        ].index.tolist()
        if len(correlated) > 0:
            corr_pairs[column] = correlated

    to_drop = []
    for col, correlated in corr_pairs.items():
        for c in correlated:
            col_to_drop = col
            if mi_scores[col] > mi_scores[c]:
                col_to_drop = c

            to_drop.append(col_to_drop)

            print(
                f"{col} and {c} are highly correlated: {round(corr_matrix.loc[col, c], 3)}. The column dropped will be {col_to_drop}."
            )

    if len(to_drop) == 0:
        print("There are no too correlated")

    return to_drop


def get_20_top_correlated_features_with_target(df, continuous_cols):
    correlations = (
        df[continuous_cols + ["TARGET"]].corr()["TARGET"].sort_values(ascending=False)
    )

    top_pos_corr = correlations[1:11]

    top_neg_corr = correlations[-10:]

    top_20_columns = pd.concat([top_pos_corr, top_neg_corr])

    plt.figure(figsize=(15, 10))
    top_20_columns.plot(kind="bar")

    plt.title("Top 20 Numerical Columns Correlated with TARGET")
    plt.xlabel("Numerical Columns")
    plt.ylabel("Correlation")
    plt.xticks()
    plt.tight_layout()
    plt.show()

    return list(top_20_columns.keys())


def calculate_statistics(dataset, column_name):
    column_data = dataset[column_name]

    mean = np.mean(column_data)
    median = np.median(column_data)
    maximum = np.nanmax(column_data)
    quantiles = np.nanquantile(column_data, [0.25, 0.5, 0.75])

    print("Mean: {:.6f}".format(mean))
    print("Median: {:.6f}".format(median))
    print("Maximum: {:.6f}".format(maximum))
    print("Quantiles:")
    print("  25% quantile: {:.6f}".format(quantiles[0]))
    print("  50% quantile: {:.6f}".format(quantiles[1]))
    print("  75% quantile: {:.6f}".format(quantiles[2]))
