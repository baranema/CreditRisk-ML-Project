# CreditRisk-ML-Project

Dataset used: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

## AnomalyGuard - Anomaly Detection.ipynb
The EDA and prediction components for building an Anomaly Detection model for financial institutions are stored in this Jupyter Notebook file. The model utilizes the IsolationForest algorithm and is subsequently deployed on GCP using Streamlit and FastAPI.

AnomalyGuard provides detection of anomalies in credit card balance datasets, enhancing fraud detection capabilities. With its adaptability to evolving fraud patterns and the ability to flag previously unseen anomalies, AnomalyGuard enables financial institutions to stay ahead of fraudsters. It empowers banks to take immediate action, protecting customer accounts, minimizing financial losses, and optimizing operational costs. AnomalyGuard is a comprehensive tool that strengthens fraud prevention efforts and enhances overall security for financial institutions.

## RiskSense - Loan Risk Clustering.ipynb
This notebook contains EDA and creation of clustering algorithm for clustering Loan Applications.

RiskSense is a risk encoding and clustering model designed to ease the assessment and categorization of applications based on their risk profiles in financial institutions. By employing advanced techniques such as data encoding using Keras and subsequent application of K-means clustering, RiskSense effectively analyzes and groups applications into low, medium, and high-risk categories. Through the capture of intricate risk patterns and the use of deep learning, RiskSense provides comprehensive understanding of application risk factors. By leveraging the power of K-means clustering, financial institutions can efficiently allocate resources and tailor their risk mitigation strategies, ultimately enhancing their risk management efforts and making informed decisions.

## LoanGuard - TARGET Prediction.ipynb
This notebook contains EDA and creation of supervised learning task to predict the target variable for loan applications, which indicates whether an application represents a client with payment difficulties (1) or all other cases (0). 

LoanGuard is a credit risk prediction model that leverages machine learning algorithms to simplify the loan application process for financial institutions. By predicting the likelihood of payment difficulties, LoanGuard enables institutions to make data-driven decisions, enhance risk management practices, and streamline loan approval processes. Through the analysis of various features and historical data, LoanGuard uncovers hidden patterns and risk factors, providing valuable insights into creditworthiness. By automating the creditworthiness assessment, LoanGuard accelerates decision-making, reduces manual effort, and improves operational efficiency. This automation leads to faster loan processing, increased customer satisfaction, optimized resource allocation, and improved overall productivity and customer experience.

## Where can these models be accessed?
Models are deployed in Google Cloud with FastAPI:
### [Backend FastAPI Application](https://backend-service-kx4s6cgoga-lm.a.run.app/)

For friendlier user interface user can upload datasets to get the predictions in frontend application:
### [Frontend Streamlit Application](https://frontend-service-kx4s6cgoga-ew.a.run.app/)
