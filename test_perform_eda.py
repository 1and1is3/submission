import pytest
from eda import perform_eda
import os

def test_perform_eda():
    perform_eda()
    assert os.path.isfile("census_formated.csv") == True