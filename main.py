# Import Union since our Item object will have tags that can be strings or
# a list.
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
from starter.ml.model import inference
from starter.ml.data import process_data
from joblib import load
from pydantic import BaseModel, Field # noqa
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cwd = os.getcwd()


# BaseModel from Pydantic is used to define data objects.

app = FastAPI()


class Value(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="Self-emp-inc")
    fnlgt: int = Field(..., example=75000)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=14, alias="education-num")
    marital_status: str = Field(...,
                                example="Married-civ-spouse",
                                alias="marital-status")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=50, alias="hours-per-week")
    native_country: str = Field(...,
                                example="United-States",
                                alias="native-country")

# This allows sending of data (our TaggedItem) via POST to the API.


@app.get("/")
async def get_items():
    return {"Welcome message!"}


@app.post("/inference")
async def get_inference(dataframe: Value):
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
    request_dict = dataframe.model_dump(by_alias=True)
    dataframe = pd.DataFrame(request_dict, index=[0])
    dataframe['salary'] = 1
    encoder = load(f'{cwd}/model/encoder.joblib')
    # Proces the test data with the process_data function.
    X_test, _, _, _ = process_data(
        dataframe, categorical_features=cat_features, label='salary',
        encoder=encoder, training=False
    )
    prediction = inference(f'{cwd}/model/model.sav', X_test)
    if prediction == 1:
        prediction = '<50k€'
    else:
        prediction = '>50k€'
    return f'{prediction}'
