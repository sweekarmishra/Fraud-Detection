from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(_name_)

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON input
        features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array
        prediction = model.predict(features)
        return jsonify({"fraud": bool(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)