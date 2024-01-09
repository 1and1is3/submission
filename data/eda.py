import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def perform_eda():
    # import and delete all spaces
    df = pd.read_csv("census_raw.csv", skipinitialspace=True)

    df.to_csv("census.csv")

if __name__ == '__main__':
    perform_eda()