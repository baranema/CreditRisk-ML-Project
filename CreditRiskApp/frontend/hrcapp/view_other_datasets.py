from __future__ import annotations

import pandas as pd
import streamlit as st

import helpers


def view_other_datasets():
    # Streamlit app
    st.title('Other Credit Risk Datasets Analysis')

    select_dataset = st.selectbox(
        'Select:', [
            'bureau_balance',
            'installments_payments',
            'bureau',
            'POS_CASH_balance',
            'previous_application',
        ],
    )

    og_df = pd.read_csv(
        f'default_data/sampled_{select_dataset}.csv', dtype={'SK_ID_CURR': int},
    )

    # Add a file uploader button
    uploaded_file = st.file_uploader(
        'Upload a CSV file or use the default credit_card_balance.csv', type='csv',
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = og_df.copy()

    # Get column types
    categorical_cols, _, continuous_cols, binary_cols = helpers.get_all_column_by_types(
        df,
    )

    if 'SK_ID_CURR' in df.columns:
        df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)

    if 'SK_ID_PREV' in df.columns:
        df['SK_ID_PREV'] = df['SK_ID_PREV'].astype(str)

    st.write('Enter an SK_ID_CURR to display matching rows')

    # Filtering
    if 'SK_ID_CURR' in df.columns:
        sk_id_curr = st.multiselect('SK_ID_CURR', list(df.SK_ID_CURR.unique()))

    if 'SK_ID_CURR' in df.columns:
        if sk_id_curr:
            curr_df = df[df['SK_ID_CURR'].isin(sk_id_curr)]
        else:
            curr_df = df
    else:
        curr_df = df

    curr_df = helpers.filter_dataframe(curr_df)
    st.dataframe(curr_df)

    # Plots
    st.write('Numerical column to plot histogram')
    selected_num_col = st.selectbox('Select:', continuous_cols)

    st.write('Show by different selected categorical value features:')
    categorical_column = st.selectbox(
        'Select:', ['Nothing Selected'] + categorical_cols + binary_cols,
    )

    if categorical_column == 'Nothing Selected':
        kde_cat_val_to_plot = None
    else:
        kde_cat_val_to_plot = categorical_column

    helpers.plot_kde(curr_df, selected_num_col, kde_cat_val_to_plot)
