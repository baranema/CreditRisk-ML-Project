from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

import helpers

API_URL = 'https://backend-service-kx4s6cgoga-lm.a.run.app/credit_balance_fraud_score/'


class SessionState:
    """Simple session state class."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_prediction(payload):
    response = requests.post(API_URL, json=payload)
    return response.status_code, response.json()


def anomaly_guard():
    # Streamlit app
    st.title('AnomalyGuard - Credit Card Balance & Fraud Analysis System')

    og_df = pd.read_csv(
        'default_data/sampled_credit_card_balance.csv', dtype={'SK_ID_CURR': int},
    )

    # Add a file uploader button
    uploaded_file = st.file_uploader(
        'Upload a CSV file or use the default credit_card_balance.csv', type='csv',
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        session_state = SessionState(first_run=True)

    else:
        df = og_df.copy()
        df = df.sample(1000, random_state=42)

    credit_card_balance = df.copy()
    credit_card_balance = credit_card_balance.fillna(
        'nan',
    ).to_dict(orient='records')
    session_state = SessionState(first_run=True)

    try:
        if session_state.first_run:
            status, predictions = get_prediction(credit_card_balance)

            df.insert(0, 'anomaly_score', '')
            df['anomaly_score'] = 'Unknown'

            df.insert(0, 'predicted_grade', '')
            df['is_fraud'] = 'Unknown'

            if status == 200:
                df['anomaly_score'] = predictions[0]['anomaly_scores']
                df['is_fraud'] = predictions[0]['is_fraud_values']
                df['is_fraud'] = df['is_fraud'].map(
                    lambda x: 1 if x == -1 else 0 if x == 1 else x,
                )

        as_quantiles = df['anomaly_score'].quantile(
            [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1],
        )

        # Get column types
        categorical_cols, _, continuous_cols, binary_cols = helpers.get_all_column_by_types(
            df,
        )

        # Move "is_fraud" column to position 2 and rename it
        df.insert(2, 'is_fraud', df.pop('is_fraud'))

        # Move "anomaly_score" column to position 3 and rename it
        df.insert(3, 'anomaly_score', df.pop('anomaly_score'))

        # Convert SK_ID_CURR and SK_ID_PREV columns to strings
        df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)
        df['SK_ID_PREV'] = df['SK_ID_PREV'].astype(str)

        st.write('Enter an SK_ID_CURR to display matching rows')

        # Filtering
        sk_id_curr = st.multiselect('SK_ID_CURR', list(df.SK_ID_CURR.unique()))

        # Display the filtered DataFrame
        if sk_id_curr:
            curr_df = df[df['SK_ID_CURR'].isin(sk_id_curr)]
        else:
            curr_df = df

        curr_df = helpers.filter_dataframe(curr_df)

        curr_df_styled = curr_df.style.applymap(
            lambda value: 'background-color: #1f692e' if value > as_quantiles.iloc[5] else
            (
                'background-color: #4b691f' if value <= as_quantiles.iloc[5] and value > as_quantiles.iloc[4] else
                (
                    'background-color: #67691f' if value <= as_quantiles.iloc[4] and value > as_quantiles.iloc[3] else
                    (
                        'background-color: #694d1f' if value <= as_quantiles.iloc[3] and value > as_quantiles.iloc[2] else
                        (
                            'background-color: #784325' if value <= as_quantiles.iloc[2] and value > as_quantiles.iloc[1] else
                            (
                                'background-color: #782525' if value <=
                                as_quantiles.iloc[1] and value > as_quantiles.iloc[0] else ''
                            )
                        )
                    )
                )
            ),
            subset=['anomaly_score'],
        )

        curr_df_styled = curr_df_styled.applymap(
            lambda value: 'color: #b03131' if value == 1 else '',
            subset=['is_fraud'],
        )

        st.dataframe(curr_df_styled)

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

    except:
        print('something has failed.')
