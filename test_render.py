import requests
import json

my_test = {
    "age": 25,
    "workclass": "Self-emp-inc",
    "fnlgt": 75000,
    "education": "Bachelors",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}
response = requests.post(
    'https://render-deployment-example-5a6q.onrender.com/inference',
    data=json.dumps(my_test))

print(response.status_code)
print(response.json())
