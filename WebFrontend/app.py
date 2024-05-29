from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('shopping_model.pkl')

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        evidence = np.array([[
            data['Administrative'], data['Administrative_Duration'], 
            data['Informational'], data['Informational_Duration'],
            data['ProductRelated'], data['ProductRelated_Duration'], 
            data['BounceRates'], data['ExitRates'], 
            data['PageValues'], data['SpecialDay'], 
            data['Month'], data['OperatingSystems'], 
            data['Browser'], data['Region'], 
            data['TrafficType'], data['VisitorType'], 
            data['Weekend']
        ]])
        prediction = model.predict(evidence)
        result = {'Revenue': bool(prediction[0])}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
