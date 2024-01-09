import os
import pytest
from eda import perform_eda

def test_perform_eda():
    perform_eda()
    assert os.path.isfile("census.csv") == True