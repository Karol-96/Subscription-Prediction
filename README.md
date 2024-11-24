# Subscription Prediction

A machine learning model to predict customer subscription churn using Logistic Regression.

## Features
- Predicts customer churn probability
- REST API using Flask
- Web interface for predictions
- Feature importance analysis

## Setup
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `python app.py`

## Model Features
The model uses the following features for prediction:
- Monthly Charges
- Total Charges
- Tenure
- Contract Type
- And more...

## API Endpoints
- GET /: Home page with prediction interface
- POST /predict: Make predictions

