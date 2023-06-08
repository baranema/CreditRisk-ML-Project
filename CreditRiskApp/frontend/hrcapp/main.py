# Import the necessary functions from the corresponding Python files
from __future__ import annotations

import streamlit as st
from client_struggle_prediction import loan_guard
from credit_card_balance import anomaly_guard
from loan_risk_cluster_prediction import risk_sense
from view_other_datasets import view_other_datasets


def main_page():
    st.markdown('# Credit Risk Analysis System')
    st.sidebar.markdown('# Credit Risk Analysis System')


def page_credit_card_balance():
    anomaly_guard()
    st.sidebar.markdown(
        '### AnomalyGuard - Credit Card Balance & Fraud Analysis',
    )


def page_other_datasets_analysis():
    view_other_datasets()
    st.sidebar.markdown('### Analysis of other datasets')


def page_risk_sense():
    risk_sense()
    st.sidebar.markdown(
        '### RiskSense - Application clustering algorithm based on their credit risk',
    )


def page_loan_guard():
    loan_guard()
    st.sidebar.markdown(
        '### LoanGuard - Client Risk Prediction Model for Loan Applications',
    )


page_names_to_funcs = {
    'Home': main_page,
    'AnomalyGuard - Credit Card Balance & Fraud Analysis': page_credit_card_balance,
    'RiskSense - Application clustering': page_risk_sense,
    'LoanGuard - Client Risk Prediction': page_loan_guard,
    'Analyse Other Datasets': page_other_datasets_analysis,
}

selected_page = st.sidebar.selectbox(
    'Select a page', page_names_to_funcs.keys(),
)
page_names_to_funcs[selected_page]()
