from fastapi import FastAPI 
from http import HTTPStatus
from credit_card_balance import CreditCardBalance
from application import Application
from application_merged import ApplicationMerged
from typing import List
import predictions as predictions
import pickle 
from tensorflow import keras  

app = FastAPI()
 
with open('models/fraud_score_detector.pkl', 'rb') as file:
    fraud_score_detector = pickle.load(file)

with open('models/cluster_assigner_preprocessor.pkl', 'rb') as file:
    cluster_assigner_preprocessor = pickle.load(file)

with open('models/cluster_assigner.pkl', 'rb') as file:
    cluster_assigner = pickle.load(file) 

with open('models/application_pca_pipeline.pkl', 'rb') as file:
    application_pca_pipeline = pickle.load(file) 

with open('models/application_preprocess.pkl', 'rb') as file:
    application_preprocess = pickle.load(file)

with open('models/target_prediction_model.pkl', 'rb') as file:
    target_prediction_model = pickle.load(file)

encoder_model = keras.models.load_model("models/encoder_model.h5")
 

@app.get("/")
def home():
    return {"message": "Bank Credit Risk Detection System", "status": HTTPStatus.OK}
 
@app.post("/credit_balance_fraud_score/")
async def predict_credit_balance_fraud_score(credit_balances: List[CreditCardBalance]):
    return predictions.predict_fraud_score(fraud_score_detector, credit_balances)

@app.post("/application_clustering/")
async def cluster_application_risk(applications: List[Application]):
    return predictions.predict_cluster(cluster_assigner_preprocessor, encoder_model, cluster_assigner, applications)

@app.post("/predict_application_struggle/")
async def predict_application_struggle(applications: List[ApplicationMerged]):
    return predictions.predict_application_struggle(application_preprocess, encoder_model, cluster_assigner, application_pca_pipeline, target_prediction_model, applications)
 