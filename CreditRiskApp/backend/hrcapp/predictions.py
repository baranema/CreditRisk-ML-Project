import pandas as pd

def predict_fraud_score(model, credit_card_balances):
    json_cc_balance = []
    for cc_balance in credit_card_balances:
        json_cc_balance.append(cc_balance.dict()) 

    df = pd.DataFrame(json_cc_balance)

    anomaly_scores = [float(val) for val in model.decision_function(df)]
    is_fraud_values = [int(val) for val in list(model.predict(df))]

    return [{
        "anomaly_scores": anomaly_scores, 
        "is_fraud_values": is_fraud_values
    }]

def predict_cluster(cluster_assigner_preprocessor, encoder_model, cluster_assigner, applications):
    json_applications = []
    for application in applications:
        json_applications.append(application.dict()) 

    df = pd.DataFrame(json_applications)
    transformed_df = cluster_assigner_preprocessor.fit_transform(df) 
    encoded_data = encoder_model.predict(transformed_df[['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT']])
    clusters = cluster_assigner.predict(encoded_data)
    clusters = [int(val) for val in list(clusters)]

    return [{
        "clusters": clusters
    }]

def predict_application_struggle(preprocessor, encoder_model, cluster_assigner, pca, model, applications):
    json_applications = []
    for application in applications:
        json_applications.append(application.dict()) 

    df = pd.DataFrame(json_applications) 
    transformed_df = preprocessor.transform(df)
    encoded_data = encoder_model.predict(transformed_df[['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT']])
    clusters = cluster_assigner.predict(encoded_data)
    transformed_df['cluster'] = clusters
    pca_results = pca.transform(transformed_df)
    predictions = model.predict(pca_results) 
    predictions = [int(val) for val in list(predictions)]
 
    return [{
        "targets": predictions
    }]