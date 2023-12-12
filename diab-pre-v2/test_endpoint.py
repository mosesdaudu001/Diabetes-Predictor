import requests

ride = {
    "Pregnancies": 2,
    "Glucose": 84,
    "BloodPressure": 0,
    "SkinThickness": 0,
    "Insulin": 0,
    "BMI": 0,
    "DiabetesPedigreeFunction": 0.304,
    "Age": 21

}

url = 'http://127.0.0.1:5000'
response = requests.post(url, json=ride)
# print(response)
print(response.json())
