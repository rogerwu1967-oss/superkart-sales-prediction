
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
import hashlib, sys, sklearn, xgboost  # For the /v1/info endpoint (deployment verification)
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model (preprocessing + final model pipeline)
MODEL_PATH = "superkart_model.joblib"
model = joblib.load(MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    MODEL_MD5 = hashlib.md5(f.read()).hexdigest()  # fingerprint of the deployed model file

# Features expected by the model (same order as in training)
FEATURES = [
    'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area', 'Product_MRP',
    'Store_Size', 'Store_Location_City_Type', 'Store_Type', 'Product_Id_char',
    'Store_Age_Years', 'Product_Type_Category'
]

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

# Information endpoint: lets us verify that the container runs the SAME model file and library versions as training
@superkart_api.get('/v1/info')
def info():
    return jsonify({
        'model_md5': MODEL_MD5,
        'model': type(model[-1]).__name__,
        'python': sys.version.split()[0],
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scikit-learn': sklearn.__version__,
        'xgboost': xgboost.__version__,
    })

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check that all required features are present
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Extract relevant features from the input data
        sample = {f: data[f] for f in FEATURES}

        # Convert the extracted data into a DataFrame
        input_data = pd.DataFrame([sample])

        # Make a prediction using the trained model
        prediction = float(model.predict(input_data)[0])

        # Return the prediction as a JSON response
        return jsonify({'Sales': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        # Get the uploaded CSV file from the request
        file = request.files['file']

        # Read the file into a DataFrame and keep only the model features
        input_data = pd.read_csv(file)[FEATURES]

        # Make predictions for the batch data
        predictions = model.predict(input_data).tolist()

        # Create an output dictionary mapping row index to predicted sales
        output_dict = {str(i): round(float(pred), 2) for i, pred in enumerate(predictions)}

        return jsonify(output_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Flask app in debug mode (only for local testing; Docker uses Gunicorn)
if __name__ == '__main__':
    superkart_api.run(debug=True)
