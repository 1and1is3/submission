# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import os
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pickle 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cwd = os.getcwd()


# Add the necessary imports for the starter code.
import pandas as pd

# Add code to load in the data.
data = pd.read_csv(f'{cwd}/../data/census.csv', index_col=0)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
    train, categorical_features=cat_features, label="salary", training=True
)
# Proces the test data with the process_data function.
print(data.head(1))
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", encoder=encoder, training=False
)
# Train and save a model.
train_model(X_train, y_train)
preds = inference('../model/model.sav', X_test)

lb = LabelBinarizer()
y_test = lb.fit_transform(y_test).ravel()


print(compute_model_metrics(y=y_test, preds=preds))
