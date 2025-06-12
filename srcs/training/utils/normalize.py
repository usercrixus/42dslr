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
    df_normalized['Biais'] = 1
    cols = ['Biais'] + [col for col in df_normalized.columns if col != 'Biais']
    df_normalized = df_normalized[cols]
    return df_normalized