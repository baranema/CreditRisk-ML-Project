from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

import helpers

API_URL = 'https://backend-service-kx4s6cgoga-lm.a.run.app/predict_application_struggle/'


class SessionState:
    """Simple session state class."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_prediction(payload):
    print('predict.')
    response = requests.post(API_URL, json=payload)
    return response.status_code, response.json()


def loan_guard():
    # Streamlit app
    st.title('LoanGuard - Client Risk/Struggle Prediction for Loan Applications')

    og_df = pd.read_csv(
        'default_data/sampled_application_merged.csv', dtype={'SK_ID_CURR': int},
    )

    # Add a file uploader button
    uploaded_file = st.file_uploader(
        'Upload a CSV file or use the default sampled_application_merged.csv', type='csv',
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        session_state = SessionState(first_run=True)

    else:
        df = og_df.copy()

    application = df.copy()
    application = application.fillna('nan').to_dict(orient='records')
    session_state = SessionState(first_run=True)

    try:
        if session_state.first_run:
            status, predictions = get_prediction(application)

            df.insert(0, 'TARGET', '')
            df['TARGET'] = 'Unknown'

            if status == 200:
                df['TARGET'] = predictions[0]['targets']

            session_state = SessionState(first_run=False)

        # Get column types
        categorical_cols, _, continuous_cols, binary_cols = helpers.get_all_column_by_types(
            df,
        )

        df.insert(2, 'TARGET', df.pop('TARGET'))

        # Convert SK_ID_CURR and SK_ID_PREV columns to strings
        df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)

        st.write('Enter an SK_ID_CURR to display matching rows')

        # Filtering
        sk_id_curr = st.multiselect('SK_ID_CURR', list(df.SK_ID_CURR.unique()))

        # Display the filtered DataFrame
        if sk_id_curr:
            curr_df = df[df['SK_ID_CURR'].isin(sk_id_curr)]
        else:
            curr_df = df

        curr_df = helpers.filter_dataframe(curr_df)

        st.dataframe(curr_df)

        # Plots
        st.write('Numerical column to plot histogram')
        selected_num_col = st.selectbox('Select:', continuous_cols)

        st.write('Show by different selected categorical value features:')
        categorical_column = st.selectbox(
            'Select:', ['Nothing Selected'] + ['TARGET'] +
            categorical_cols + binary_cols,
        )

        if categorical_column == 'Nothing Selected':
            kde_cat_val_to_plot = None
        else:
            kde_cat_val_to_plot = categorical_column

        helpers.plot_kde(curr_df, selected_num_col, kde_cat_val_to_plot)

    except:
        print('something has failed.')
