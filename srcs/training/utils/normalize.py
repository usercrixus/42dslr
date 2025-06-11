import pandas as pd

def normalize(file):

    df = pd.read_csv(file)
    df = df.drop(columns=[
        'Index',
        'First Name',
        'Last Name',
        'Best Hand',
        'Hogwarts House',
        'Birthday'
    ], errors='ignore')

    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized