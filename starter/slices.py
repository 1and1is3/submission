from ml.data import process_data
from ml.model import inference, compute_model_metrics
from joblib import load
import os
from sklearn.preprocessing import LabelBinarizer
cwd = os.getcwd()

slice_marital_status = [
    'Never-married',
    'Married-civ-spouse',
    'Divorced',
    'Married-spouse-absent',
    'Separated',
    'Married-AF-spouse',
    'Widowed']


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


def get_data(data):
    encoder = load(f'{cwd}/../../model/encoder.joblib')

    for value in slice_marital_status:
        sliced = data[(data['marital-status'] == value)]

        X_test, y_test, _, _ = process_data(
            sliced, categorical_features=cat_features, label='salary',
            encoder=encoder, training=False)
        preds = inference('../model/model.sav', X_test)

        lb = LabelBinarizer()
        y_test = lb.fit_transform(y_test).ravel()
        print(f'{value}: {compute_model_metrics(y=y_test, preds=preds)}')
        with open('../slice_output.txt', 'a') as f:
            f.write(
                f'{value}: {compute_model_metrics(y=y_test, preds=preds)}\n'
                )
