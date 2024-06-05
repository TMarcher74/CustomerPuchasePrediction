from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load all four models
model1 = joblib.load('KNN.pkl')
model2 = joblib.load('NBC.pkl')
model3 = joblib.load('SVM.pkl')
model4 = joblib.load('RF.pkl')

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received data: {data}")  # Logging the received data
        
        # Convert received data into numpy array
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
        
        # Predict using all four models
        prediction1 = model1.predict(evidence)
        prediction2 = model2.predict(evidence)
        prediction3 = model3.predict(evidence)
        prediction4 = model4.predict(evidence)
        
        # Prepare the result
        result = {
            'Model1_Revenue': bool(prediction1[0]),
            'Model2_Revenue': bool(prediction2[0]),
            'Model3_Revenue': bool(prediction3[0]),
            'Model4_Revenue': bool(prediction4[0])
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)