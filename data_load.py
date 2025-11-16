import pandas as pd

def load_data():
    dta = pd.read_csv('../data/complaints.csv')
    return dta

