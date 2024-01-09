import pytest
from submission.data.eda import perform_eda
import os

def test_perform_eda():
    perform_eda()
    assert os.path.isfile("../data/census.csv") == True