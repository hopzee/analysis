import pandas as pd

def load_data():
    data = pd.read_csv("dataset.csv")
    return data