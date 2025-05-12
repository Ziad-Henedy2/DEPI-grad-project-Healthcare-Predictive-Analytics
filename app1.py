from flask import Flask, request, jsonify, send_from_directory
import sqlite3
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load pre-trained ensemble model with error handling
try:
    model = joblib.load('random_forest_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set to None if loading fails

# Database Setup
def init_db():
    try:
        conn = sqlite3.connect('patients.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS patients 
                     (id INTEGER PRIMARY KEY, prediction TEXT, date TEXT, data TEXT)''')
        conn.commit()
        print("Database initialized")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

# Serve the HTML file
@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

# Updated predict route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.json
        print(f"Received data: {data}")  # Debug output
        
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Define numeric and categorical columns
        numeric_cols = [
            'age', 'time_in_hospital', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'num_procedures', 'num_medications', 'number_diagnoses'
        ]
        categorical_cols = [
            'race', 'gender', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'max_glu_serum', 'A1Cresult', 'metformin',
            'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'tolazamide', 'insulin', 'glyburide-metformin', 'diag_1'
        ]
        
        # Convert numeric columns to integers
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Apply one-hot encoding to categorical columns
        df_encoded = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])
        
        # Get the model's expected feature names
        try:
            expected_features = model.feature_names_in_  # Requires scikit-learn >= 1.0
        except AttributeError:
            # Fallback: Define expected features based on input data and common categories
            expected_features = [
                'age', 'time_in_hospital', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'num_procedures', 'num_medications', 'number_diagnoses',
                'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Other',
                'gender_Female', 'gender_Male',
                'admission_type_id_1', 'admission_type_id_2', 'admission_type_id_3',
                'admission_type_id_4', 'admission_type_id_5', 'admission_type_id_6',
                'admission_type_id_7', 'admission_type_id_8',
                'discharge_disposition_id_1', 'discharge_disposition_id_2', 'discharge_disposition_id_3',
                'discharge_disposition_id_4', 'discharge_disposition_id_5', 'discharge_disposition_id_6',
                'discharge_disposition_id_7', 'discharge_disposition_id_8', 'discharge_disposition_id_9',
                'discharge_disposition_id_10', 'discharge_disposition_id_11', 'discharge_disposition_id_12',
                'discharge_disposition_id_13', 'discharge_disposition_id_14', 'discharge_disposition_id_15',
                'discharge_disposition_id_16', 'discharge_disposition_id_17', 'discharge_disposition_id_18',
                'discharge_disposition_id_19', 'discharge_disposition_id_20', 'discharge_disposition_id_21',
                'discharge_disposition_id_22', 'discharge_disposition_id_23', 'discharge_disposition_id_24',
                'discharge_disposition_id_25', 'discharge_disposition_id_26', 'discharge_disposition_id_27',
                'discharge_disposition_id_28', 'discharge_disposition_id_29', 'discharge_disposition_id_30',
                'admission_source_id_1', 'admission_source_id_2', 'admission_source_id_3',
                'admission_source_id_4', 'admission_source_id_5', 'admission_source_id_6',
                'admission_source_id_7', 'admission_source_id_8', 'admission_source_id_9',
                'admission_source_id_10', 'admission_source_id_11', 'admission_source_id_12',
                'admission_source_id_13', 'admission_source_id_14', 'admission_source_id_15',
                'admission_source_id_17', 'admission_source_id_18', 'admission_source_id_19',
                'admission_source_id_20', 'admission_source_id_21', 'admission_source_id_22',
                'admission_source_id_23', 'admission_source_id_24', 'admission_source_id_25',
                'admission_source_id_26',
                'max_glu_serum_>200', 'max_glu_serum_>300', 'max_glu_serum_None', 'max_glu_serum_Norm',
                'A1Cresult_>7', 'A1Cresult_>8', 'A1Cresult_None', 'A1Cresult_Norm',
                'metformin_Down', 'metformin_No', 'metformin_Steady', 'metformin_Up',
                'repaglinide_Down', 'repaglinide_No', 'repaglinide_Steady', 'repaglinide_Up',
                'nateglinide_Down', 'nateglinide_No', 'nateglinide_Steady', 'nateglinide_Up',
                'chlorpropamide_Down', 'chlorpropamide_No', 'chlorpropamide_Steady', 'chlorpropamide_Up',
                'glimepiride_Down', 'glimepiride_No', 'glimepiride_Steady', 'glimepiride_Up',
                'glipizide_Down', 'glipizide_No', 'glipizide_Steady', 'glipizide_Up',
                'glyburide_Down', 'glyburide_No', 'glyburide_Steady', 'glyburide_Up',
                'pioglitazone_Down', 'pioglitazone_No', 'pioglitazone_Steady', 'pioglitazone_Up',
                'rosiglitazone_Down', 'rosiglitazone_No', 'rosiglitazone_Steady', 'rosiglitazone_Up',
                'acarbose_Down', 'acarbose_No', 'acarbose_Steady', 'acarbose_Up',
                'tolazamide_Down', 'tolazamide_No', 'tolazamide_Steady', 'tolazamide_Up',
                'insulin_Down', 'insulin_No', 'insulin_Steady', 'insulin_Up',
                'glyburide-metformin_Down', 'glyburide-metformin_No', 'glyburide-metformin_Steady', 'glyburide-metformin_Up'
                # Note: diag_1 excluded here; handled separately below
            ]
        
        # Handle diag_1 (ICD codes) separately
        # Assume model expects specific ICD codes or a simplified representation
        # For simplicity, create a binary column for the provided diag_1 value (e.g., diag_1_205)
        if 'diag_1' in df.columns:
            diag_value = df['diag_1'].iloc[0]
            df_encoded[f'diag_1_{diag_value}'] = 1
            # Remove original diag_1 column if still present
            if 'diag_1' in df_encoded.columns:
                df_encoded = df_encoded.drop(columns=['diag_1'])
        
        # Align DataFrame with expected features
        for col in expected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_features]
        
        # Debugging: Print columns to verify
        print("Encoded DataFrame columns:", df_encoded.columns.tolist())
        
        # Make prediction
        prediction = model.predict(df_encoded)[0]
        probability = max(model.predict_proba(df_encoded)[0]) * 100
        
        # Save to database
        conn = sqlite3.connect('patients.db')
        c = conn.cursor()
        c.execute("INSERT INTO patients (prediction, date, data) VALUES (?, ?, ?)", 
                  (str(prediction), datetime.now().strftime('%Y-%m-%d'), str(data)))
        conn.commit()
        conn.close()
        
        return jsonify({'prediction': str(prediction), 'probability': probability})
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

# Dashboard route (unchanged)
@app.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        conn = sqlite3.connect('patients.db')
        c = conn.cursor()
        c.execute("SELECT id, prediction, date FROM patients")
        rows = [{'id': row[0], 'prediction': row[1], 'date': row[2]} for row in c.fetchall()]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        print(f"Error in /dashboard: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)