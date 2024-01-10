from main import app
from fastapi.testclient import TestClient
import json


client = TestClient(app)


def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == ['Welcome message!']


def test_inference():
    data = {
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
    response = client.post("/inference/", data=json.dumps(data))
    assert response.status_code == 200


def test_inference_request():
    data = {
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
    response = client.post("/inference/", data=json.dumps(data))
    assert response.json() == ('>50k€' or '<50k€')


def test_inference_request_2():
    data = {
        "age": 11,
        "workclass": "Self-emp-inc",
        "fnlgt": 4,
        "education": "Bachelors",
        "education-num": 4,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 1,
        "native-country": "United-States"
    }
    response = client.post("/inference/", data=json.dumps(data))
    assert response.json() == ('>50k€' or '<50k€')
