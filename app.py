from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(_name)  # Fixed __name_

# Path to the trained model
MODEL_PATH = "fraud_detection_model.pkl"

# Load the trained model if it exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    model = None
    print("❌ Error: Model file not found!")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not found. Please upload fraud_detection_model.pkl."})

    try:
        data = request.json  # Expecting JSON input
        features = np.array(data["features"]).reshape(1, -1)  # Convert input to NumPy array
        prediction = model.predict(features)  # Make prediction
        return jsonify({"fraud": bool(prediction[0])})  # Return prediction as JSON

    except Exception as e:
        return jsonify({"error": str(e)})  # Handle errors

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
