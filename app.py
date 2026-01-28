import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- Load Model and Scaler ---

model = load_model('rainfall_model.h5') # Changed from .keras to .h5
scaler = joblib.load('scaler.pkl')

# Define features order (MUST match your training data)
# 1. Humidity, 2. Soil Moisture, 3. Temperature, 4. Rainfall
FEATURES = ['humidity_percent', 'soil_moisture_m3m3', 'temperature_c', 'rainfall_mm']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        # Expected format: A list of 30 days of data. 
        # Each day is [humidity, moisture, temp, rainfall]
        data = request.json['data'] 
        
        # Convert to numpy array
        input_data = np.array(data)
        
        # Validation: check shape
        if input_data.shape != (30, 4):
            return jsonify({'error': f'Expected shape (30, 4), got {input_data.shape}'}), 400

        # --- Preprocessing ---
        # 1. Scale the data using the loaded scaler
        scaled_data = scaler.transform(input_data)
        
        # 2. Reshape for LSTM: (1 sample, 30 time steps, 4 features)
        lstm_input = scaled_data.reshape(1, 30, 4)
        
        # --- Prediction ---
        prediction_scaled = model.predict(lstm_input)
        
        # --- Inverse Scaling ---
        # We need to inverse transform to get the value in mm.
        # The scaler expects 4 columns, but prediction is just 1 value (rainfall).
        # We create a dummy matrix to trick the scaler.
        dummy_matrix = np.zeros((1, 4))
        # Place prediction in the rainfall column (index 3)
        dummy_matrix[0, 3] = prediction_scaled[0][0]
        
        # Inverse transform and extract the rainfall value
        prediction_actual = scaler.inverse_transform(dummy_matrix)[0, 3]
        
        # --- Interpret Result ---
        # Simple logic for the advisory system
        status = "No Rain"
        if prediction_actual > 0.5: status = "Light Rain"
        if prediction_actual > 10.0: status = "Moderate Rain"
        if prediction_actual > 30.0: status = "Heavy Rain"

        return jsonify({
            'prediction_mm': float(round(prediction_actual, 2)),
            'status': status,
            'advisory': get_advisory(status)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_advisory(status):
    if status == "Heavy Rain":
        return "Delay irrigation. Ensure drainage channels are clear."
    elif status == "Moderate Rain":
        return "Irrigation may not be needed. Monitor soil moisture."
    elif status == "Light Rain":
        return "Light irrigation recommended if soil moisture is low."
    else:
        return "Standard irrigation schedule applies."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
