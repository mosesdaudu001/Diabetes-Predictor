from flask import Flask,request, jsonify
import numpy as np
import pandas as pd
import pickle

application=Flask(__name__)

app=application

## Route for a home page
# Remove Random addition
serverless_classifier = pickle.load(open('diabetes_svm_model.pkl', 'rb'))
serverless_scaler = pickle.load(open('scaler_min_max.pkl', 'rb'))

@app.route('/',methods=['POST'])
def predict_datapoint():
  
  request_json = request.get_json()

  Pregnancies = request_json['Pregnancies']
  Glucose = request_json['Glucose']
  BloodPressure = request_json['BloodPressure']
  SkinThickness = request_json['SkinThickness']
  Insulin = request_json['Insulin']
  BMI = request_json['BMI']
  DiabetesPedigreeFunction = request_json['DiabetesPedigreeFunction']
  Age = request_json['Age']
  row_values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

  x_new = np.array(row_values).reshape(1, -1)
  x_new_scaler = serverless_scaler.transform(x_new)
  y_new_pred = serverless_classifier.predict(x_new_scaler)

  result = {
     "prediction": int(y_new_pred[0])
  }
#   return "The prediction is {}".format(jsonify(prediction))
  return jsonify(result)
    

if __name__=="__main__":
    app.run(debug=True)        
