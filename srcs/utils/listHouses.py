import pandas as pd

df = pd.read_csv("datasets/dataset_train.csv", usecols=["Hogwarts House"])
print(df["Hogwarts House"].dropna().unique())
