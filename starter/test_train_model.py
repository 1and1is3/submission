import pandas as pd
# process_data : y and x same shape
# model.py : train model exist model.sav
# inference : imput shape same as output shape
from joblib import load
from starter.ml.model import train_model, inference
from starter.ml.data import process_data
import os
cwd = os.getcwd()


def test_process_data():
    data = pd.read_csv(f'{cwd}/../data/census.csv', index_col=0)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True)

    assert X_train.shape[0] == data.shape[0]


def test_train_model():
    data = pd.read_csv(f'{cwd}/../data/census.csv', index_col=0)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    train_model(X_train, y_train)
    assert os.path.isfile("../model/model.sav")


def test_inference():
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    encoder = load(f'{cwd}/../model/encoder.joblib')
    data = pd.read_csv(f'{cwd}/../data/census.csv', index_col=0)
    X_test, y_test, _, _ = process_data(
        data, categorical_features=cat_features, encoder=encoder,
        label="salary", training=True
    )

    preds = inference('../model/model.sav', X_test)
    assert preds.shape[0] == X_test.shape[0]
