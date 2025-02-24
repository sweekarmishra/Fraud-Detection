Fraud Detection API:
📌 Overview
This is a Flask-based API for fraud detection in credit card transactions. The model predicts whether a transaction is fraudulent based on input features.
🚀 Features

-Loads a trained machine learning model (fraud_detection_model.pkl).
-Accepts JSON input via a /predict endpoint.
-Returns a fraud prediction (true/false) in JSON format.
-Deployed on Render/Heroku for easy access.

🛠 Tech Stack
-Backend: Flask
-Machine Learning: Scikit-learn, NumPy, Pandas
-Deployment: Render

📂 Project Structure
fraud-detection-api/
│── app.py                  # Main Flask application
│── requirements.txt        # Dependencies
│── fraud_detection_model.pkl  # Trained ML model
│── Procfile                # For deployment
│── README.md               # Project documentation
│── .gitignore              # Ignore unnecessary files

📦 Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/yourusername/fraud-detection-api.git
cd fraud-detection-api

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Flask App
python app.py

The API will be available at: http://127.0.0.1:5000/

🔮 API Usage
Endpoint: /predict (POST Request)

📌 Input JSON Format

{
  "features": [0.5, 1.2, 3.4, ..., 2.1]
}

📌 Response Format

{
  "fraud": true
}

🌐 Deployment (Render/Heroku)

For deployment, ensure:
1. Procfile contains:
web: gunicorn app:app
2. Use requirements.txt for dependencies.

💡 Future Improvements
-Improve model accuracy with more training data.
-Add authentication for secure access.


