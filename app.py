from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the fraud detection model
model_path = "fraud_detection_model.pkl"  # Ensure this file exists
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Home route
@app.route("/")
def home():
    return render_template("index.html")  # Ensure "templates/index.html" exists

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert data to NumPy array
        features = np.array(data["features"]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)

        return jsonify({"prediction": int(prediction[0])})  # Return prediction

    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Handle errors

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
