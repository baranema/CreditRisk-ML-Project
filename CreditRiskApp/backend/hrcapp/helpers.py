from __future__ import annotations

import matplotlib.pylab as plt
import pandas as pd
from sklearn import pipeline as sklearn_pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


class AutomaticColumnTransformer(ColumnTransformer):
    def __init__(
        self,
        *,
        remainder='drop',
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
        continuous_cols, categorical_cols, categorical_num_cols, binary_cols = self._get_column_types(
            X[self.selected_cols],
        )
        self.transformers = self.get_transformers(
            continuous_cols, categorical_cols, categorical_num_cols, binary_cols,
        )

        return super().fit_transform(X, y)

    def _get_column_types(self, data):
        cat_cols, cat_num_cols, num_cols, binary_cols = get_all_column_by_types(
            data, target_cols='TARGET',
        )
        return num_cols, cat_cols, cat_num_cols, binary_cols

    def get_transformers(self, continuous_cols, categorical_cols, categorical_num_cols, binary_cols):
        num_transformer = sklearn_pipeline.Pipeline(
            [
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
            ],
        )

        cat_num_transformer = sklearn_pipeline.Pipeline(
            [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder()),
            ],
        )

        binary_transformer = sklearn_pipeline.Pipeline(
            [
                ('imputer', SimpleImputer(strategy='most_frequent')),
            ],
        )

        cat_transformer = sklearn_pipeline.Pipeline(
            [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ],
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, continuous_cols),
                ('binary', binary_transformer, binary_cols),
                ('cat', cat_transformer, categorical_cols),
                ('cat_num', cat_num_transformer, categorical_num_cols),
            ],
            remainder=self.remainder,
        )

        preprocessor.set_output(transform='pandas')

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

        if not hasattr(self, '_sklearn_output_config'):
            self._sklearn_output_config = {}

        self._sklearn_output_config['transform'] = transform
        return self


def rename_columns(X):
    original_column_names = []
    for col in X.columns:
        if '__' in col:
            new_col = col.split('__')
            original_column_names.append(new_col[1])
        else:
            original_column_names.append(col)

    X.columns = original_column_names
    return X


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
                                lambda x: {
                                    'Y': 1, 'N': 0,
                                    'Yes': 1, 'No': 0,
                                }.get(x, x),
                            )
                            binary_cols.append(col)
                        else:
                            categorical_cols.append(col)

    return categorical_cols, categorical_numerical_cols, continuous_cols, binary_cols


def create_new_application_features(data):
    df = data.copy()
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df


def get_application_cleaner(
    df,
    missing_patterns,
    fill_by_None,
    replace_by_zero,
):

    fill_Unaccompanied = [
        col for col in df.columns if 'Unaccompanied' in list(df[col].unique())
    ]
    replace_with_OTHER = [
        col for col in df.columns if 'OTHER' in list(df[col].unique())
    ]
    replace_by_No = [
        col for col in df.columns if 'No' in list(df[col].unique())
    ]
    columns_to_impute = missing_patterns

    mean_imputer = SimpleImputer(strategy='mean')
    constant_imputer = SimpleImputer(strategy='constant', fill_value=-999)

    final_preprocessor = sklearn_pipeline.Pipeline([])
    i = 0

    already_imputed_cols = []
    for columns_to_impute in missing_patterns:
        missing_pipeline = sklearn_pipeline.Pipeline(
            [
                ('missing_mask', MissingMaskTransformer(columns_to_impute)),
                ('imputer', constant_imputer),
                ('imputer_none', mean_imputer),
                (
                    'columns_renamer',
                    preprocessing.FunctionTransformer(rename_columns),
                ),
            ],
        )

        pattern_imputer = ColumnTransformer(
            transformers=[
                ('pattern_impute', missing_pipeline, columns_to_impute),
            ],
            remainder='passthrough',
        )

        pattern_imputer.set_output(transform='pandas')

        pipeline = sklearn_pipeline.Pipeline(
            [
                ('pattern_imputer', pattern_imputer),
                (
                    'columns_renamed_1',
                    preprocessing.FunctionTransformer(rename_columns),
                ),
            ],
        )
        final_preprocessor.steps.append(
            (f'missing_cols_cleaner_{i}', pipeline),
        )
        i += 1
        already_imputed_cols.extend(columns_to_impute)

    already_imputed_cols = list(set(already_imputed_cols))
    categorical_cols, categorical_numerical_cols, continuous_cols, binary_cols = get_all_column_by_types(
        df,
    )
    replace_by_zero = list(
        {col for col in replace_by_zero if col not in already_imputed_cols},
    )

    all_cat_cols = list(
        set(categorical_cols + categorical_numerical_cols + binary_cols),
    )
    continuous_cols = list(
        {
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
        },
    )
    all_cat_cols_for_impute = list(
        {
            col
            for col in all_cat_cols
            if col not in fill_Unaccompanied
            and col not in fill_by_None
            and col not in replace_with_OTHER
            and col not in replace_by_No
            and col not in replace_by_zero
            and col not in continuous_cols
            and col not in already_imputed_cols
        },
    )

    constant_imputer = ColumnTransformer(
        transformers=[
            (
                'imputer_Unaccompanied',
                SimpleImputer(strategy='constant', fill_value='Unaccompanied'),
                [
                    col for col in list(set(fill_Unaccompanied))
                    if col not in already_imputed_cols
                ],
            ),
            (
                'imputer_None',
                SimpleImputer(strategy='constant', fill_value='None'),
                [
                    col for col in list(set(fill_by_None))
                    if col not in already_imputed_cols
                ],
            ),
            (
                'imputer_OTHER',
                SimpleImputer(strategy='constant', fill_value='OTHER'),
                [
                    col for col in list(set(replace_with_OTHER))
                    if col not in already_imputed_cols
                ],
            ),
            (
                'imputer_No',
                SimpleImputer(strategy='constant', fill_value='No'),
                [
                    col for col in list(set(replace_by_No))
                    if col not in already_imputed_cols
                ],
            ),
            (
                'imputer_Zero',
                SimpleImputer(strategy='constant', fill_value=0),
                [
                    col for col in list(set(replace_by_zero))
                    if col not in already_imputed_cols
                ],
            ),
            ('imputer_Mean', SimpleImputer(strategy='mean'), continuous_cols),
            (
                'imputer_Mode',
                SimpleImputer(strategy='most_frequent'),
                all_cat_cols_for_impute,
            ),
        ],
        remainder='passthrough',
    )
    constant_imputer.set_output(transform='pandas')

    pipeline = sklearn_pipeline.Pipeline(
        [
            ('pattern_imputer', constant_imputer),
            ('columns_renamed_2', preprocessing.FunctionTransformer(rename_columns)),
        ],
    )

    final_preprocessor.steps.append((f'cleaner_imputer', pipeline))

    pipeline = sklearn_pipeline.Pipeline(
        [
            (
                'feature_engineering',
                preprocessing.FunctionTransformer(
                    create_new_application_features,
                ),
            ),
        ],
    )

    final_preprocessor.steps.append(('new_feature_creation', pipeline))

    return final_preprocessor


def get_final_application_preprocessor_pipeline(
    cleaner, selected_cols, others_drop='drop',
):
    final_preprocessor = sklearn_pipeline.Pipeline([])
    final_preprocessor.steps.append((f'cleaner', cleaner))

    selected_cols_with_new = [
        'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT',
        'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT',
    ] + selected_cols

    preprocessor = AutomaticColumnTransformer(
        remainder=others_drop, selected_cols=selected_cols_with_new,
    )
    preprocessor.set_output(transform='pandas')
    final_preprocessor.steps.append((f'final_preprocessor', preprocessor))

    final_preprocessor.steps.append(
        ('prefixes_removal', preprocessing.FunctionTransformer(rename_columns)),
    )

    return final_preprocessor
