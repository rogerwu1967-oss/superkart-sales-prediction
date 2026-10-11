
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model (preprocessing + XGBoost pipeline)
model = joblib.load("superkart_model.joblib")

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
#        prediction = float(model.predict(input_data)[0])   Roger  Mask

raw_prediction = float(model.predict(input_data)[0])

if raw_prediction < 0:
    superkart_api.logger.warning(
        "Negative sales prediction: %.2f; input=%s",
        raw_prediction, sample
    )

prediction = max(0.0, raw_prediction)
return jsonify({"Sales": round(prediction, 2)})


raw_predictions = model.predict(input_data).astype(float)

negative_count = int((raw_predictions < 0).sum())
if negative_count:
    superkart_api.logger.warning(
        "Clipped %d negative batch predictions", negative_count
    )

predictions = np.maximum(raw_predictions, 0.0)
output_dict = {
    str(i): round(float(pred), 2)
    for i, pred in enumerate(predictions)
}
return jsonify(output_dict)
# Roger end
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
