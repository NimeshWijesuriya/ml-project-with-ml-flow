from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.mlProject.pipeline.prediction import PredictionPipeline
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home_page():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def train():
    try:
        train_model()
        return "Training Successful!"
    except Exception as e:
        return f"Training failed: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and validate inputs
        gender = str(request.form.get('Customer Gender', ''))
        SeniorCitizen = int(request.form.get('Senior Citizen', 0))  # Convert to integer
        Partner = str(request.form.get('Partner', ''))
        Dependents = str(request.form.get('Dependents', ''))
        tenure = int(request.form.get('Network Stay', 0))  # Convert to integer
        PhoneService = str(request.form.get('Value Added Service', ''))
        MultipleLines = str(request.form.get('Multiple Lines', ''))
        InternetService = str(request.form.get('Internet Service', ''))
        OnlineSecurity = str(request.form.get('Online Security', ''))
        OnlineBackup = str(request.form.get('Online Backup', ''))
        DeviceProtection = str(request.form.get('Device Protection', ''))
        TechSupport = str(request.form.get('Tech Support', ''))
        StreamingTV = str(request.form.get('Streaming TV', ''))
        StreamingMovies = str(request.form.get('Streaming Movies', ''))
        Contract = str(request.form.get('Contract', ''))
        PaperlessBilling = str(request.form.get('Paperless Billing', ''))
        PaymentMethod = str(request.form.get('Payment Method', ''))
        MonthlyCharges = float(request.form.get('Monthly Charges', 0.0))  # Convert to float
        TotalCharges = str(request.form.get('Total Charges', '0.0'))  # Keep as string
        
        
        
        # Define categorical columns and possible values
        categorical_columns = {
            'gender': ['Female', 'Male'],
            'Partner': ['No', 'Yes'],
            'Dependents': ['No', 'Yes'],
            'PhoneService': ['No', 'Yes'],
            'MultipleLines': ['No', 'No phone service', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['No', 'No internet service', 'Yes'],
            'OnlineBackup': ['No', 'No internet service', 'Yes'],
            'DeviceProtection': ['No', 'No internet service', 'Yes'],
            'TechSupport': ['No', 'No internet service', 'Yes'],
            'StreamingTV': ['No', 'No internet service', 'Yes'],
            'StreamingMovies': ['No', 'No internet service', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['No', 'Yes'],
            'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
            'tenure_group': ['tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36', 'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72']
        }
        
        # Create a DataFrame from the input data
        input_data = {
            'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        }
        df = pd.DataFrame(input_data)
        def tenure_group(tenure):
            if 1 <= tenure <= 12:
                return 'tenure_group_1 - 12'
            elif 13 <= tenure <= 24:
                return 'tenure_group_13 - 24'
            elif 25 <= tenure <= 36:
                return 'tenure_group_25 - 36'
            elif 37 <= tenure <= 48:
                return 'tenure_group_37 - 48'
            elif 49 <= tenure <= 60:
                return 'tenure_group_49 - 60'
            elif 61 <= tenure <= 72:
                return 'tenure_group_61 - 72'
            else:
                return 'tenure_group_61 - 72'

# Applying the function to create a new column
        df['tenure_group'] = df['tenure'].apply(tenure_group)

        df_dummies = pd.get_dummies(df, columns=categorical_columns.keys(), drop_first=False)
        all_possible_columns = []
        for col, categories in categorical_columns.items():
            all_possible_columns.extend([f"{col}_{category}" for category in categories])
        final_df = df_dummies.reindex(columns=all_possible_columns, fill_value=0)
        
        non_categorical_columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']
        final_df = pd.concat([df[non_categorical_columns], final_df], axis=1)
        data = final_df.replace({True: 1, False: 0})

        data = np.array(data)
        
        # Predict
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(data)
        result = "[Customer Will Churn]" if prediction == 1 else "[Customer Will Not Churn]"
        
        return render_template('result.html', prediction=result)

    except Exception as e:
        import traceback
        error_message = f"Prediction error: {e}\n{traceback.format_exc()}"
        print(error_message)
        return f"An error occurred during prediction: {error_message}"

def train_model():
    # Placeholder for actual training code
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
