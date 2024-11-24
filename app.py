from flask import Flask,request, render_template,jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


app = Flask(__name__)

#Loading the saved model
model = joblib.load('logistic_model.pkl')

#Defining the features names in correct order
feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
                 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

@app.route('/')
def home():
    return render_template('./templates/index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        #Getting data from form
        features = {}
        for feature in feature_names:
            value = request.form.get(feature)
            features[feature] = float(value) if value.replace('.','').isdigit() else value

        #Create a dataframe
        df = pd.DataFrame([features])

        #Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify(
            {
                          'prediction': int(prediction),
            'probability': float(probability),
            'message': 'Customer is likely to churn' if prediction == 1 else 'Customer is likely to stay'  
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
